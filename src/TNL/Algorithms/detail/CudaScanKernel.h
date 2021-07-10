/***************************************************************************
                          CudaScanKernel.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/Math.h>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Containers/Array.h>

namespace TNL {
namespace Algorithms {
namespace detail {

#ifdef HAVE_CUDA

template< typename Real,
          typename Reduction,
          typename Index >
__global__ void
cudaFirstPhaseBlockScan( const ScanType scanType,
                         Reduction reduction,
                         const Real zero,
                         const Index size,
                         const int elementsInBlock,
                         const Real* input,
                         Real* output,
                         Real* blockResults )
{
   Real* sharedData = TNL::Cuda::getSharedMemory< Real >();
   Real* auxData = &sharedData[ elementsInBlock + elementsInBlock / Cuda::getNumberOfSharedMemoryBanks() + 2 ];
   Real* warpSums = &auxData[ blockDim.x ];

   const Index lastElementIdx = size - blockIdx.x * elementsInBlock;
   const Index lastElementInBlock = TNL::min( lastElementIdx, elementsInBlock );

   /***
    * Load data into the shared memory.
    */
   const int blockOffset = blockIdx.x * elementsInBlock;
   int idx = threadIdx.x;
   if( scanType == ScanType::Exclusive )
   {
      if( idx == 0 )
         sharedData[ 0 ] = zero;
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ Cuda::getInterleaving( idx + 1 ) ] = input[ blockOffset + idx ];
         idx += blockDim.x;
      }
   }
   else
   {
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ Cuda::getInterleaving( idx ) ] = input[ blockOffset + idx ];
         idx += blockDim.x;
      }
   }

   /***
    * Perform the sequential prefix-sum.
    */
   __syncthreads();
   const int chunkSize = elementsInBlock / blockDim.x;
   const int chunkOffset = threadIdx.x * chunkSize;
   const int numberOfChunks = roundUpDivision( lastElementInBlock, chunkSize );

   if( chunkOffset < lastElementInBlock )
   {
      auxData[ threadIdx.x ] =
         sharedData[ Cuda::getInterleaving( chunkOffset ) ];
   }

   int chunkPointer = 1;
   while( chunkPointer < chunkSize &&
          chunkOffset + chunkPointer < lastElementInBlock )
   {
      sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer ) ] =
         reduction( sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer ) ],
                    sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ] );
      auxData[ threadIdx.x ] =
         sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer ) ];
      chunkPointer++;
   }

   /***
    *  Perform the parallel prefix-sum inside warps.
    */
   const int threadInWarpIdx = threadIdx.x % Cuda::getWarpSize();
   const int warpIdx = threadIdx.x / Cuda::getWarpSize();
   for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
      if( threadInWarpIdx >= stride && threadIdx.x < numberOfChunks )
         auxData[ threadIdx.x ] = reduction( auxData[ threadIdx.x ], auxData[ threadIdx.x - stride ] );
      __syncwarp();
   }

   if( threadInWarpIdx == Cuda::getWarpSize() - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx.x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
         if( threadInWarpIdx >= stride )
            warpSums[ threadIdx.x ] = reduction( warpSums[ threadIdx.x ], warpSums[ threadIdx.x - stride ] );
         __syncwarp();
      }
   __syncthreads();

   /****
    * Shift the warp prefix-sums.
    */
   if( warpIdx > 0 )
      auxData[ threadIdx.x ] = reduction( auxData[ threadIdx.x ], warpSums[ warpIdx - 1 ] );
   __syncthreads();

   /***
    *  Store the result back in global memory.
    */
   idx = threadIdx.x;
   while( idx < elementsInBlock && blockOffset + idx < size )
   {
      const int chunkIdx = idx / chunkSize;
      Real chunkShift( zero );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      sharedData[ Cuda::getInterleaving( idx ) ] =
         reduction( sharedData[ Cuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ Cuda::getInterleaving( idx ) ];
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( scanType == ScanType::Exclusive )
      {
         blockResults[ blockIdx.x ] = reduction( sharedData[ Cuda::getInterleaving( lastElementInBlock - 1 ) ],
                                                 sharedData[ Cuda::getInterleaving( lastElementInBlock ) ] );
      }
      else
         blockResults[ blockIdx.x ] = sharedData[ Cuda::getInterleaving( lastElementInBlock - 1 ) ];
   }
}

template< typename Real,
          typename Reduction,
          typename Index >
__global__ void
cudaSecondPhaseBlockScan( Reduction reduction,
                          const Index size,
                          const int elementsInBlock,
                          const Index gridIdx,
                          const Index maxGridSize,
                          const Real* blockResults,
                          Real* data,
                          Real shift )
{
   shift = reduction( shift, blockResults[ gridIdx * maxGridSize + blockIdx.x ] );
   const int readOffset = blockIdx.x * elementsInBlock;
   int readIdx = threadIdx.x;
   while( readIdx < elementsInBlock && readOffset + readIdx < size )
   {
      data[ readIdx + readOffset ] = reduction( data[ readIdx + readOffset ], shift );
      readIdx += blockDim.x;
   }
}

template< ScanType scanType,
          typename Real,
          typename Index >
struct CudaScanKernelLauncher
{
   /****
    * \brief Performs both phases of prefix sum.
    *
    * \param size  Number of elements to be scanned.
    * \param deviceInput  Pointer to input data on GPU.
    * \param deviceOutput  Pointer to output array on GPU, can be the same as input.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param zero  Neutral element for given reduction operation, i.e. value such that
    *              `reduction(zero, x) == x` for any `x`.
    * \param blockSize  The CUDA block size to be used for kernel launch.
    */
   template< typename Reduction >
   static void
   perform( const Index size,
            const Real* deviceInput,
            Real* deviceOutput,
            Reduction& reduction,
            const Real zero,
            const int blockSize = 256 )
   {
      const auto blockShifts = performFirstPhase(
         size,
         deviceInput,
         deviceOutput,
         reduction,
         zero,
         blockSize );
      performSecondPhase(
         size,
         deviceOutput,
         blockShifts.getData(),
         reduction,
         zero,
         blockSize );
   }

   /****
    * \brief Performs the first phase of prefix sum.
    *
    * \param size  Number of elements to be scanned.
    * \param deviceInput  Pointer to input data on GPU.
    * \param deviceOutput  Pointer to output array on GPU, can be the same as input.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param zero  Neutral value for given reduction operation, i.e. value such that
    *              `reduction(zero, x) == x` for any `x`.
    * \param blockSize  The CUDA block size to be used for kernel launch.
    */
   template< typename Reduction >
   static auto
   performFirstPhase( const Index size,
                      const Real* deviceInput,
                      Real* deviceOutput,
                      Reduction& reduction,
                      const Real zero,
                      const int blockSize = 256 )
   {
      // compute the number of grids
      const int elementsInBlock = 8 * blockSize;
      const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
      const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );
      //std::cerr << "numberOfgrids =  " << numberOfGrids << std::endl;

      // allocate array for the block results
      Containers::Array< Real, Devices::Cuda > blockResults;
      blockResults.setSize( numberOfBlocks + 1 );
      blockResults.setElement( 0, zero );

      // loop over all grids
      for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
         // compute current grid size and size of data to be scanned
         const Index gridOffset = gridIdx * maxGridSize() * elementsInBlock;
         Index currentSize = size - gridOffset;
         if( currentSize / elementsInBlock > maxGridSize() )
            currentSize = maxGridSize() * elementsInBlock;
         //std::cerr << "GridIdx = " << gridIdx << " grid size = " << currentSize << std::endl;

         // setup block and grid size
         dim3 cudaBlockSize, cudaGridSize;
         cudaBlockSize.x = blockSize;
         cudaGridSize.x = roundUpDivision( currentSize, elementsInBlock );

         // run the kernel
         const std::size_t sharedDataSize = elementsInBlock +
                                            elementsInBlock / Cuda::getNumberOfSharedMemoryBanks() + 2;
         const std::size_t sharedMemory = ( sharedDataSize + blockSize + Cuda::getWarpSize() ) * sizeof( Real );
         cudaFirstPhaseBlockScan<<< cudaGridSize, cudaBlockSize, sharedMemory >>>
            ( scanType,
              reduction,
              zero,
              currentSize,
              elementsInBlock,
              &deviceInput[ gridOffset ],
              &deviceOutput[ gridOffset ],
              // blockResults are shifted by 1, because the 0-th element should stay zero
              &blockResults.getData()[ gridIdx * maxGridSize() + 1 ] );
      }

      // synchronize the null-stream after all grids
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;

      // blockResults now contains scan results for each block. The first phase
      // ends by computing an exclusive scan of this array.
      if( numberOfBlocks > 1 ) {
         // we perform an inclusive scan, but the 0-th is zero and block results
         // were shifted by 1, so effectively we get an exclusive scan
         CudaScanKernelLauncher< ScanType::Inclusive, Real, Index >::perform(
            blockResults.getSize(),
            blockResults.getData(),
            blockResults.getData(),
            reduction,
            zero,
            blockSize );
      }

      // Store the number of CUDA grids for the purpose of unit testing, i.e.
      // to check if we test the algorithm with more than one CUDA grid.
      gridsCount() = numberOfGrids;

      // blockResults now contains shift values for each block - to be used in the second phase
      return blockResults;
   }

   /****
    * \brief Performs the second phase of prefix sum.
    *
    * \param size  Number of elements to be scanned.
    * \param deviceOutput  Pointer to output array on GPU.
    * \param blockShifts  Pointer to a GPU array containing the block shifts. It is the
    *                     result of the first phase.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param shift  A constant shifting all elements of the array (usually `zero`, i.e.
    *               the neutral value).
    * \param blockSize  The CUDA block size to be used for kernel launch.
    */
   template< typename Reduction >
   static void
   performSecondPhase( const Index size,
                       Real* deviceOutput,
                       const Real* blockShifts,
                       Reduction& reduction,
                       const Real shift,
                       const Index blockSize = 256 )
   {
      // compute the number of grids
      const int elementsInBlock = 8 * blockSize;
      const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
      const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      // loop over all grids
      for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
         // compute current grid size and size of data to be scanned
         const Index gridOffset = gridIdx * maxGridSize() * elementsInBlock;
         Index currentSize = size - gridOffset;
         if( currentSize / elementsInBlock > maxGridSize() )
            currentSize = maxGridSize() * elementsInBlock;
         //std::cerr << "GridIdx = " << gridIdx << " grid size = " << currentSize << std::endl;

         // setup block and grid size
         dim3 cudaBlockSize, cudaGridSize;
         cudaBlockSize.x = blockSize;
         cudaGridSize.x = roundUpDivision( currentSize, elementsInBlock );

         // run the kernel
         cudaSecondPhaseBlockScan<<< cudaGridSize, cudaBlockSize >>>
            ( reduction,
              size,
              elementsInBlock,
              gridIdx,
              (Index) maxGridSize(),
              blockShifts,
              &deviceOutput[ gridOffset ],
              shift );
      }

      // synchronize the null-stream after all grids
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;
   }

   // The following serves for setting smaller maxGridSize so that we can force
   // the scan in CUDA to run with more than one grid in unit tests.
   static int& maxGridSize()
   {
      static int maxGridSize = Cuda::getMaxGridSize();
      return maxGridSize;
   }

   static void resetMaxGridSize()
   {
      maxGridSize() = Cuda::getMaxGridSize();
   }

   static int& gridsCount()
   {
      static int gridsCount = -1;
      return gridsCount;
   }
};

#endif

} // namespace detail
} // namespace Algorithms
} // namespace TNL
