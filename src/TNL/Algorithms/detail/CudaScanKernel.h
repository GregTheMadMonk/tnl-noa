/***************************************************************************
                          CudaScanKernel.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Math.h>
#include <TNL/Cuda/SharedMemory.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Containers/Array.h>
#include "ScanType.h"

namespace TNL {
namespace Algorithms {
namespace detail {

#ifdef HAVE_CUDA

template< ScanType scanType,
          int blockSize,
          int valuesPerThread,
          typename InputView,
          typename OutputView,
          typename Reduction >
__global__ void
CudaScanKernelFirstPhase( const InputView input,
                          OutputView output,
                          typename InputView::IndexType begin,
                          typename InputView::IndexType end,
                          typename OutputView::IndexType outputBegin,
                          Reduction reduction,
                          typename OutputView::ValueType zero,
                          typename OutputView::ValueType* blockResults )
{
   using ValueType = typename OutputView::ValueType;
   using IndexType = typename InputView::IndexType;

   // verify the configuration
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelFirstPhase" );
   static_assert( blockSize / Cuda::getWarpSize() <= Cuda::getWarpSize(),
                  "blockSize is too large, it would not be possible to scan warpResults using one warp" );

   // calculate indices
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
   const int elementsInBlock = TNL::min( remainingElements, maxElementsInBlock );

   // update global array offsets for the thread
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   begin += threadOffset;
   outputBegin += threadOffset;

   // allocate shared memory
   constexpr int shmemElements = maxElementsInBlock + maxElementsInBlock / Cuda::getNumberOfSharedMemoryBanks() + 2;
   __shared__ ValueType sharedData[ shmemElements ];  // accessed via Cuda::getInterleaving()
   __shared__ ValueType chunkResults[ blockSize ];
   __shared__ ValueType warpResults[ Cuda::getWarpSize() ];

   /***
    * Load data into the shared memory.
    */
   int idx = threadIdx.x;
   if( scanType == ScanType::Exclusive )
   {
      if( idx == 0 )
         sharedData[ 0 ] = zero;
      while( idx < elementsInBlock )
      {
         sharedData[ Cuda::getInterleaving( idx + 1 ) ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
   }
   else
   {
      while( idx < elementsInBlock )
      {
         sharedData[ Cuda::getInterleaving( idx ) ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
   }
   __syncthreads();

   /***
    * Perform the sequential scan of the chunk in shared memory.
    */
   const int chunkOffset = threadIdx.x * valuesPerThread;
   const int numberOfChunks = roundUpDivision( elementsInBlock, valuesPerThread );

   int chunkPointer = 1;
   while( chunkPointer < valuesPerThread && chunkOffset + chunkPointer < elementsInBlock )
   {
      sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer ) ] =
         reduction( sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer ) ],
                    sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ] );
      chunkPointer++;
   }

   // store the result of the sequential reduction of the chunk in chunkResults
   chunkResults[ threadIdx.x ] = sharedData[ Cuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ];
   __syncthreads();

   /***
    * Perform the parallel scan on chunkResults inside warps.
    */
   const int threadInWarpIdx = threadIdx.x % Cuda::getWarpSize();
   const int warpIdx = threadIdx.x / Cuda::getWarpSize();
   for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
      if( threadInWarpIdx >= stride && threadIdx.x < numberOfChunks )
         chunkResults[ threadIdx.x ] = reduction( chunkResults[ threadIdx.x ], chunkResults[ threadIdx.x - stride ] );
      __syncwarp();
   }

   if( threadInWarpIdx == Cuda::getWarpSize() - 1 )
      warpResults[ warpIdx ] = chunkResults[ threadIdx.x ];
   __syncthreads();

   /****
    * Perform the scan of warpResults using one warp.
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
         if( threadInWarpIdx >= stride )
            warpResults[ threadIdx.x ] = reduction( warpResults[ threadIdx.x ], warpResults[ threadIdx.x - stride ] );
         __syncwarp();
      }
   __syncthreads();

   /****
    * Shift chunkResults by the warpResults.
    */
   if( warpIdx > 0 )
      chunkResults[ threadIdx.x ] = reduction( chunkResults[ threadIdx.x ], warpResults[ warpIdx - 1 ] );
   __syncthreads();

   /***
    * Store the result back in global memory.
    */
   idx = threadIdx.x;
   while( idx < elementsInBlock )
   {
      const int chunkIdx = idx / valuesPerThread;
      ValueType chunkShift = zero;
      if( chunkIdx > 0 )
         chunkShift = chunkResults[ chunkIdx - 1 ];
      output[ outputBegin ] =
      sharedData[ Cuda::getInterleaving( idx ) ] =
         reduction( sharedData[ Cuda::getInterleaving( idx ) ], chunkShift );
      outputBegin += blockDim.x;
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( scanType == ScanType::Exclusive )
      {
         blockResults[ blockIdx.x ] = reduction( sharedData[ Cuda::getInterleaving( elementsInBlock - 1 ) ],
                                                 sharedData[ Cuda::getInterleaving( elementsInBlock ) ] );
      }
      else
         blockResults[ blockIdx.x ] = sharedData[ Cuda::getInterleaving( elementsInBlock - 1 ) ];
   }
}

template< int blockSize,
          int valuesPerThread,
          typename OutputView,
          typename Reduction >
__global__ void
CudaScanKernelSecondPhase( OutputView output,
                           typename OutputView::IndexType outputBegin,
                           typename OutputView::IndexType outputEnd,
                           Reduction reduction,
                           int gridOffset,
                           const typename OutputView::ValueType* blockResults,
                           typename OutputView::ValueType shift )
{
   // load the block result into a __shared__ variable first
   __shared__ typename OutputView::ValueType blockResult;
   if( threadIdx.x == 0 )
      blockResult = blockResults[ gridOffset + blockIdx.x ];

   // update the output offset for the thread
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelFirstPhase" );
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   outputBegin += threadOffset;

   // update the block shift
   __syncthreads();
   shift = reduction( shift, blockResult );

   int valueIdx = 0;
   while( valueIdx < valuesPerThread && outputBegin < outputEnd )
   {
      output[ outputBegin ] = reduction( output[ outputBegin ], shift );
      outputBegin += blockDim.x;
      valueIdx++;
   }
}

/**
 * \tparam blockSize  The CUDA block size to be used for kernel launch.
 * \tparam valuesPerThread  Number of elements processed by each thread sequentially.
 */
template< ScanType scanType,
          int blockSize = 256,
          int valuesPerThread = 8 >
struct CudaScanKernelLauncher
{
   /****
    * \brief Performs both phases of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param zero  Neutral element for given reduction operation, i.e. value such that
    *              `reduction(zero, x) == x` for any `x`.
    */
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static void
   perform( const InputArray& input,
            OutputArray& output,
            typename InputArray::IndexType begin,
            typename InputArray::IndexType end,
            typename OutputArray::IndexType outputBegin,
            Reduction&& reduction,
            typename OutputArray::ValueType zero )
   {
      const auto blockShifts = performFirstPhase(
         input,
         output,
         begin,
         end,
         outputBegin,
         reduction,
         zero );
      performSecondPhase(
         input,
         output,
         blockShifts,
         begin,
         end,
         outputBegin,
         reduction,
         zero );
   }

   /****
    * \brief Performs the first phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param zero  Neutral value for given reduction operation, i.e. value such that
    *              `reduction(zero, x) == x` for any `x`.
    */
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static auto
   performFirstPhase( const InputArray& input,
                      OutputArray& output,
                      typename InputArray::IndexType begin,
                      typename InputArray::IndexType end,
                      typename OutputArray::IndexType outputBegin,
                      Reduction&& reduction,
                      typename OutputArray::ValueType zero )
   {
      using Index = typename InputArray::IndexType;

      // compute the number of grids
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const Index numberOfBlocks = roundUpDivision( end - begin, maxElementsInBlock );
      const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      // allocate array for the block results
      Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
      blockResults.setSize( numberOfBlocks + 1 );
      blockResults.setElement( 0, zero );

      // loop over all grids
      for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
         // compute current grid offset and size of data to be scanned
         const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
         const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

         // setup block and grid size
         dim3 cudaBlockSize, cudaGridSize;
         cudaBlockSize.x = blockSize;
         cudaGridSize.x = roundUpDivision( currentSize, maxElementsInBlock );

         // run the kernel
         CudaScanKernelFirstPhase< scanType, blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
            ( input.getConstView(),
              output.getView(),
              begin + gridOffset,
              begin + gridOffset + currentSize,
              outputBegin + gridOffset,
              reduction,
              zero,
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
         CudaScanKernelLauncher< ScanType::Inclusive >::perform(
            blockResults,
            blockResults,
            0,
            blockResults.getSize(),
            0,
            reduction,
            zero );
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
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param blockShifts  Pointer to a GPU array containing the block shifts. It is the
    *                     result of the first phase.
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction  Symmetric binary function representing the reduction operation
    *                   (usually addition, i.e. an instance of \ref std::plus).
    * \param shift  A constant shifting all elements of the array (usually `zero`, i.e.
    *               the neutral value).
    */
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( const InputArray& input,
                       OutputArray& output,
                       const BlockShifts& blockShifts,
                       typename InputArray::IndexType begin,
                       typename InputArray::IndexType end,
                       typename OutputArray::IndexType outputBegin,
                       Reduction&& reduction,
                       typename OutputArray::ValueType zero )
   {
      using Index = typename InputArray::IndexType;

      // compute the number of grids
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const Index numberOfBlocks = roundUpDivision( end - begin, maxElementsInBlock );
      const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      // loop over all grids
      for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
         // compute current grid offset and size of data to be scanned
         const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
         const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

         // setup block and grid size
         dim3 cudaBlockSize, cudaGridSize;
         cudaBlockSize.x = blockSize;
         cudaGridSize.x = roundUpDivision( currentSize, maxElementsInBlock );

         // run the kernel
         CudaScanKernelSecondPhase< blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
            ( output.getView(),
              outputBegin + gridOffset,
              outputBegin + gridOffset + currentSize,
              reduction,
              gridIdx * maxGridSize(),
              blockShifts.getData(),
              zero );
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
      gridsCount() = -1;
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
