/***************************************************************************
                          CudaPrefixSumKernel.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/Math.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef HAVE_CUDA

template< typename Real,
          typename Reduction,
          typename Index >
__global__ void
cudaFirstPhaseBlockPrefixSum( const PrefixSumType prefixSumType,
                              Reduction reduction,
                              const Real zero,
                              const Index size,
                              const Index elementsInBlock,
                              const Real* input,
                              Real* output,
                              Real* auxArray,
                              const Real gridShift )
{
   Real* sharedData = TNL::Devices::Cuda::getSharedMemory< Real >();
   Real* auxData = &sharedData[ elementsInBlock + elementsInBlock / Devices::Cuda::getNumberOfSharedMemoryBanks() + 2 ];
   Real* warpSums = &auxData[ blockDim.x ];

   const Index lastElementIdx = size - blockIdx.x * elementsInBlock;
   const Index lastElementInBlock = TNL::min( lastElementIdx, elementsInBlock );

   /***
    * Load data into the shared memory.
    */
   const Index blockOffset = blockIdx.x * elementsInBlock;
   Index idx = threadIdx.x;
   if( prefixSumType == PrefixSumType::Exclusive )
   {
      if( idx == 0 )
         sharedData[ 0 ] = zero;
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ Devices::Cuda::getInterleaving( idx + 1 ) ] = input[ blockOffset + idx ];
         idx += blockDim.x;
      }
   }
   else
   {
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ Devices::Cuda::getInterleaving( idx ) ] = input[ blockOffset + idx ];
         idx += blockDim.x;
      }
   }
   if( blockIdx.x == 0 && threadIdx.x == 0 )
      sharedData[ 0 ] = reduction( sharedData[ 0 ], gridShift );

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
         sharedData[ Devices::Cuda::getInterleaving( chunkOffset ) ];
   }

   Index chunkPointer( 1 );
   while( chunkPointer < chunkSize &&
          chunkOffset + chunkPointer < lastElementInBlock )
   {
      sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer ) ] =
         reduction( sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer ) ],
                    sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ] );
      auxData[ threadIdx.x ] =
         sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer ) ];
      chunkPointer++;
   }

   /***
    *  Perform the parallel prefix-sum inside warps.
    */
   const int threadInWarpIdx = threadIdx.x % Devices::Cuda::getWarpSize();
   const int warpIdx = threadIdx.x / Devices::Cuda::getWarpSize();
   for( int stride = 1; stride < Devices::Cuda::getWarpSize(); stride *= 2 ) {
      if( threadInWarpIdx >= stride && threadIdx.x < numberOfChunks )
         auxData[ threadIdx.x ] = reduction( auxData[ threadIdx.x ], auxData[ threadIdx.x - stride ] );
      __syncwarp();
   }

   if( threadInWarpIdx == Devices::Cuda::getWarpSize() - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx.x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < Devices::Cuda::getWarpSize(); stride *= 2 ) {
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
      const Index chunkIdx = idx / chunkSize;
      Real chunkShift( zero );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      sharedData[ Devices::Cuda::getInterleaving( idx ) ] =
         reduction( sharedData[ Devices::Cuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ Devices::Cuda::getInterleaving( idx ) ];
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( prefixSumType == PrefixSumType::Exclusive )
      {
         Real aux = zero;
         aux = reduction( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ] );
         aux = reduction( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock ) ] );
         auxArray[ blockIdx.x ] = aux;
      }
      else
         auxArray[ blockIdx.x ] = sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ];
   }
}

template< typename Real,
          typename Reduction,
          typename Index >
__global__ void
cudaSecondPhaseBlockPrefixSum( Reduction reduction,
                               const Index size,
                               const Index elementsInBlock,
                               Real gridShift,
                               const Real* auxArray,
                               Real* data )
{
   if( blockIdx.x > 0 )
   {
      const Real shift = auxArray[ blockIdx.x - 1 ];
      const Index readOffset = blockIdx.x * elementsInBlock;
      Index readIdx = threadIdx.x;
      while( readIdx < elementsInBlock && readOffset + readIdx < size )
      {
         data[ readIdx + readOffset ] = reduction( data[ readIdx + readOffset ], shift );
         readIdx += blockDim.x;
      }
   }
}

template< PrefixSumType prefixSumType,
          typename Real,
          typename Index >
struct CudaPrefixSumKernelLauncher
{
   template< typename Reduction >
   static void
   cudaRecursivePrefixSum( PrefixSumType prefixSumType_,
                           Reduction& reduction,
                           const Real& zero,
                           const Index size,
                           const Index blockSize,
                           const Index elementsInBlock,
                           Real& gridShift,
                           const Real* input,
                           Real* output )
   {
      const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
      const Index auxArraySize = numberOfBlocks;

      Array< Real, Devices::Cuda > auxArray1, auxArray2;
      auxArray1.setSize( auxArraySize );
      auxArray2.setSize( auxArraySize );

      /****
       * Setup block and grid size.
       */
      dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
      cudaBlockSize.x = blockSize;
      cudaGridSize.x = roundUpDivision( size, elementsInBlock );

      /****
       * Run the kernel.
       */
      const std::size_t sharedDataSize = elementsInBlock +
                                         elementsInBlock / Devices::Cuda::getNumberOfSharedMemoryBanks() + 2;
      const std::size_t sharedMemory = ( sharedDataSize + blockSize + Devices::Cuda::getWarpSize() ) * sizeof( Real );
      cudaFirstPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize, sharedMemory >>>
         ( prefixSumType_,
           reduction,
           zero,
           size,
           elementsInBlock,
           input,
           output,
           auxArray1.getData(),
           gridShift );
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;


      //std::cerr << " auxArray1 = " << auxArray1 << std::endl;
      /***
       * In auxArray1 there is now a sum of numbers in each block.
       * We must compute prefix-sum of auxArray1 and then shift
       * each block.
       */
      Real gridShift2 = zero;
      if( numberOfBlocks > 1 )
         cudaRecursivePrefixSum( PrefixSumType::Inclusive,
            reduction,
            zero,
            numberOfBlocks,
            blockSize,
            elementsInBlock,
            gridShift2,
            auxArray1.getData(),
            auxArray2.getData() );

      //std::cerr << " auxArray2 = " << auxArray2 << std::endl;
      cudaSecondPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize >>>
         ( reduction,
           size,
           elementsInBlock,
           gridShift,
           auxArray2.getData(),
           output );
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;

      cudaMemcpy( &gridShift,
                  &auxArray2[ auxArraySize - 1 ],
                  sizeof( Real ),
                  cudaMemcpyDeviceToHost );
      //std::cerr << "gridShift = " << gridShift << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }

   /****
    * \brief Starts prefix sum in CUDA.
    *
    * \tparam Reduction reduction to be performed on particular elements - addition usually
    * \param size is number of elements to be scanned
    * \param blockSize is CUDA block size
    * \param deviceInput is pointer to input data on GPU
    * \param deviceOutput is pointer to resulting array, can be the same as input
    * \param reduction is instance of Reduction
    * \param zero is neutral element for given Reduction
    */
   template< typename Reduction >
   static void
   start( const Index size,
          const Index blockSize,
          const Real *deviceInput,
          Real* deviceOutput,
          Reduction& reduction,
          const Real& zero )
   {
      /****
       * Compute the number of grids
       */
      const Index elementsInBlock = 8 * blockSize;
      const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
      //const auto maxGridSize = 3; //Devices::Cuda::getMaxGridSize();
      const Index numberOfGrids = Devices::Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize );
      Real gridShift = zero;
      //std::cerr << "numberOfgrids =  " << numberOfGrids << std::endl;

      /****
       * Loop over all grids.
       */
      for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ )
      {
         /****
          * Compute current grid size and size of data to be scanned
          */
         const Index gridOffset = gridIdx * maxGridSize * elementsInBlock;
         Index currentSize = size - gridOffset;
         if( currentSize / elementsInBlock > maxGridSize )
            currentSize = maxGridSize * elementsInBlock;

         //std::cerr << "GridIdx = " << gridIdx << " grid size = " << currentSize << std::endl;
         cudaRecursivePrefixSum( prefixSumType,
            reduction,
            zero,
            currentSize,
            blockSize,
            elementsInBlock,
            gridShift,
            &deviceInput[ gridOffset ],
            &deviceOutput[ gridOffset ] );
      }

      /***
       * Store the number of CUDA grids for the purpose of unit testing, i.e.
       * to check if we test the algorithm with more than one CUDA grid.
       */
      gridsCount = numberOfGrids;
   }

   /****
    * The following serves for setting smaller maxGridSize so that we can force
    * the prefix sum in CUDA to run with more the one grids in unit tests.
    */
   static void setMaxGridSize( int newMaxGridSize ) {
      maxGridSize = newMaxGridSize;
   }

   static void resetMaxGridSize() {
      maxGridSize = Devices::Cuda::getMaxGridSize();
   }

   static int maxGridSize;

   static int gridsCount;
};

template< PrefixSumType prefixSumType,
          typename Real,
          typename Index >
int CudaPrefixSumKernelLauncher< prefixSumType, Real, Index >::maxGridSize = Devices::Cuda::getMaxGridSize();

template< PrefixSumType prefixSumType,
          typename Real,
          typename Index >
int CudaPrefixSumKernelLauncher< prefixSumType, Real, Index >::gridsCount = -1;


#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
