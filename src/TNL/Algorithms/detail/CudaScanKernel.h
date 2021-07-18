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
   constexpr int shmemElements = maxElementsInBlock + maxElementsInBlock / Cuda::getNumberOfSharedMemoryBanks();
   __shared__ ValueType sharedData[ shmemElements ];  // accessed via Cuda::getInterleaving()
   __shared__ ValueType chunkResults[ blockSize + blockSize / Cuda::getNumberOfSharedMemoryBanks() ];  // accessed via Cuda::getInterleaving()
   __shared__ ValueType warpResults[ Cuda::getWarpSize() ];

   // Load data into the shared memory.
   {
      int idx = threadIdx.x;
      while( idx < elementsInBlock )
      {
         sharedData[ Cuda::getInterleaving( idx ) ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
      // fill the remaining (maxElementsInBlock - elementsInBlock) values with zero
      // (this helps to avoid divergent branches in the blocks below)
      while( idx < maxElementsInBlock )
      {
         sharedData[ Cuda::getInterleaving( idx ) ] = zero;
         idx += blockDim.x;
      }
   }
   __syncthreads();

   // Perform sequential reduction of the chunk in shared memory.
   const int chunkOffset = threadIdx.x * valuesPerThread;
   const int chunkResultIdx = Cuda::getInterleaving( threadIdx.x );
   {
      ValueType chunkResult = sharedData[ Cuda::getInterleaving( chunkOffset ) ];
      #pragma unroll
      for( int i = 1; i < valuesPerThread; i++ )
         chunkResult = reduction( chunkResult, sharedData[ Cuda::getInterleaving( chunkOffset + i ) ] );

      // store the result of the sequential reduction of the chunk in chunkResults
      chunkResults[ chunkResultIdx ] = chunkResult;
   }
   __syncthreads();

   // Perform the parallel scan on chunkResults inside warps.
   const int threadInWarpIdx = threadIdx.x % Cuda::getWarpSize();
   const int warpIdx = threadIdx.x / Cuda::getWarpSize();
   #pragma unroll
   for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
      if( threadInWarpIdx >= stride ) {
         chunkResults[ chunkResultIdx ] = reduction( chunkResults[ chunkResultIdx ], chunkResults[ Cuda::getInterleaving( threadIdx.x - stride ) ] );
      }
      __syncwarp();
   }

   // The last thread in warp stores the intermediate result in warpResults.
   if( threadInWarpIdx == Cuda::getWarpSize() - 1 )
      warpResults[ warpIdx ] = chunkResults[ chunkResultIdx ];
   __syncthreads();

   // Perform the scan of warpResults using one warp.
   if( warpIdx == 0 )
      #pragma unroll
      for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
         if( threadInWarpIdx >= stride )
            warpResults[ threadIdx.x ] = reduction( warpResults[ threadIdx.x ], warpResults[ threadIdx.x - stride ] );
         __syncwarp();
      }
   __syncthreads();

   // Shift chunkResults by the warpResults.
   if( warpIdx > 0 )
      chunkResults[ chunkResultIdx ] = reduction( chunkResults[ chunkResultIdx ], warpResults[ warpIdx - 1 ] );
   __syncthreads();

   // Downsweep step: scan the chunks and use the chunk result as the initial value.
   {
      ValueType value = zero;
      if( threadIdx.x > 0 )
         value = chunkResults[ Cuda::getInterleaving( threadIdx.x - 1 ) ];

      #pragma unroll
      for( int i = 0; i < valuesPerThread; i++ )
      {
         const int sharedIdx = Cuda::getInterleaving( chunkOffset + i );
         const ValueType inputValue = sharedData[ sharedIdx ];
         if( scanType == ScanType::Exclusive )
            sharedData[ sharedIdx ] = value;
         value = reduction( value, inputValue );
         if( scanType == ScanType::Inclusive )
            sharedData[ sharedIdx ] = value;
      }

      // The last thread of the block stores the block result in the global memory.
      if( blockResults && threadIdx.x == blockDim.x - 1 )
         blockResults[ blockIdx.x ] = value;
   }
   __syncthreads();

   // Store the result back in the global memory.
   {
      int idx = threadIdx.x;
      while( idx < elementsInBlock )
      {
         output[ outputBegin ] = sharedData[ Cuda::getInterleaving( idx ) ];
         outputBegin += blockDim.x;
         idx += blockDim.x;
      }
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
