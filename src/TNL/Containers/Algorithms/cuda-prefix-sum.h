/***************************************************************************
                          cuda-prefix-sum.h  -  description
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
#include <TNL/Containers/Algorithms/ReductionOperations.h>
#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

enum class PrefixSumType
{
   exclusive,
   inclusive
};

#ifdef HAVE_CUDA

template< typename DataType,
          typename Operation,
          typename VolatileOperation,
          typename Index >
__global__ void
cudaFirstPhaseBlockPrefixSum( const PrefixSumType prefixSumType,
                              Operation operation,
                              VolatileOperation volatileOperation,
                              const DataType zero,
                              const Index size,
                              const Index elementsInBlock,
                              const DataType* input,
                              DataType* output,
                              DataType* auxArray )
{
   DataType* sharedData = TNL::Devices::Cuda::getSharedMemory< DataType >();
   volatile DataType* auxData = &sharedData[ elementsInBlock + elementsInBlock / Devices::Cuda::getNumberOfSharedMemoryBanks() + 2 ];
   volatile DataType* warpSums = &auxData[ blockDim.x ];

   const Index lastElementIdx = size - blockIdx.x * elementsInBlock;
   const Index lastElementInBlock = TNL::min( lastElementIdx, elementsInBlock );

   /***
    * Load data into the shared memory.
    */
   const Index blockOffset = blockIdx.x * elementsInBlock;
   Index idx = threadIdx.x;
   if( prefixSumType == PrefixSumType::exclusive )
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
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ Devices::Cuda::getInterleaving( idx ) ] = input[ blockOffset + idx ];
         idx += blockDim.x;
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
         sharedData[ Devices::Cuda::getInterleaving( chunkOffset ) ];
   }

   Index chunkPointer( 1 );
   while( chunkPointer < chunkSize &&
          chunkOffset + chunkPointer < lastElementInBlock )
   {
      operation( sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer ) ],
                 sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ] );
      auxData[ threadIdx.x ] =
         sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer  ) ];
      chunkPointer++;
   }

   /***
    *  Perform the parallel prefix-sum inside warps.
    */
   const int threadInWarpIdx = threadIdx.x % Devices::Cuda::getWarpSize();
   const int warpIdx = threadIdx.x / Devices::Cuda::getWarpSize();
   for( int stride = 1; stride < Devices::Cuda::getWarpSize(); stride *= 2 )
      if( threadInWarpIdx >= stride && threadIdx.x < numberOfChunks )
         volatileOperation( auxData[ threadIdx.x ], auxData[ threadIdx.x - stride ] );

   if( threadInWarpIdx == Devices::Cuda::getWarpSize() - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx.x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < Devices::Cuda::getWarpSize(); stride *= 2 )
         if( threadInWarpIdx >= stride )
            volatileOperation( warpSums[ threadIdx.x ], warpSums[ threadIdx.x - stride ] );
   __syncthreads();

   /****
    * Shift the warp prefix-sums.
    */
   if( warpIdx > 0 )
      volatileOperation( auxData[ threadIdx.x ], warpSums[ warpIdx - 1 ] );

   /***
    *  Store the result back in global memory.
    */
   __syncthreads();
   idx = threadIdx.x;
   while( idx < elementsInBlock && blockOffset + idx < size )
   {
      const Index chunkIdx = idx / chunkSize;
      DataType chunkShift( zero );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      operation( sharedData[ Devices::Cuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ Devices::Cuda::getInterleaving( idx ) ];
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( prefixSumType == PrefixSumType::exclusive )
      {
         DataType aux = zero;
         operation( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ] );
         operation( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock ) ] );
         auxArray[ blockIdx.x ] = aux;
      }
      else
         auxArray[ blockIdx.x ] = sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ];
   }
}

template< typename DataType,
          typename Operation,
          typename Index >
__global__ void
cudaSecondPhaseBlockPrefixSum( Operation operation,
                               const Index size,
                               const Index elementsInBlock,
                               DataType gridShift,
                               const DataType* auxArray,
                               DataType* data )
{
   if( blockIdx.x > 0 )
   {
      operation( gridShift, auxArray[ blockIdx.x - 1 ] );

      const Index readOffset = blockIdx.x * elementsInBlock;
      Index readIdx = threadIdx.x;
      while( readIdx < elementsInBlock && readOffset + readIdx < size )
      {
         operation( data[ readIdx + readOffset ], gridShift );
         readIdx += blockDim.x;
      }
   }
}

template< typename DataType,
          typename Operation,
          typename VolatileOperation,
          typename Index >
void
cudaRecursivePrefixSum( const PrefixSumType prefixSumType,
                        Operation& operation,
                        VolatileOperation& volatileOperation,
                        const DataType& zero,
                        const Index size,
                        const Index blockSize,
                        const Index elementsInBlock,
                        const DataType gridShift,
                        const DataType* input,
                        DataType *output )
{
   const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
   const Index auxArraySize = numberOfBlocks * sizeof( DataType );

   Array< DataType, Devices::Cuda > auxArray1, auxArray2;
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
   const std::size_t sharedMemory = ( sharedDataSize + blockSize + Devices::Cuda::getWarpSize()  ) * sizeof( DataType );
   cudaFirstPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize, sharedMemory >>>
      ( prefixSumType,
        operation,
        volatileOperation,
        zero,
        size,
        elementsInBlock,
        input,
        output,
        auxArray1.getData() );
   TNL_CHECK_CUDA_DEVICE;

   /***
    * In auxArray1 there is now a sum of numbers in each block.
    * We must compute prefix-sum of auxArray1 and then shift
    * each block.
    */
   if( numberOfBlocks > 1 )
       cudaRecursivePrefixSum( PrefixSumType::inclusive,
                               operation,
                               volatileOperation,
                               zero,
                               numberOfBlocks,
                               blockSize,
                               elementsInBlock,
                               gridShift,
                               auxArray1.getData(),
                               auxArray2.getData() );

   cudaSecondPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize >>>
      ( operation,
        size,
        elementsInBlock,
        gridShift,
        auxArray2.getData(),
        output );
   TNL_CHECK_CUDA_DEVICE;
}


template< typename DataType,
          typename Operation,
          typename VolatileOperation,
          typename Index >
void
cudaGridPrefixSum( PrefixSumType prefixSumType,
                   Operation& operation,
                   VolatileOperation& volatileOperation,
                   const DataType& zero,
                   const Index size,
                   const Index blockSize,
                   const Index elementsInBlock,
                   const DataType *deviceInput,
                   DataType *deviceOutput,
                   DataType& gridShift )
{
   cudaRecursivePrefixSum( prefixSumType,
                           operation,
                           volatileOperation,
                           zero,
                           size,
                           blockSize,
                           elementsInBlock,
                           gridShift,
                           deviceInput,
                           deviceOutput );

   cudaMemcpy( &gridShift,
               &deviceOutput[ size - 1 ],
               sizeof( DataType ),
               cudaMemcpyDeviceToHost );
   TNL_CHECK_CUDA_DEVICE;
}

/////
// deviceInput and deviceOutput can be the same
template< typename DataType,
          typename Operation,
          typename VolatileOperation,
          typename Index >
void
cudaPrefixSum( const Index size,
               const Index blockSize,
               const DataType *deviceInput,
               DataType* deviceOutput,
               Operation& operation,
               VolatileOperation& volatileOperation,
               const DataType& zero,
               const PrefixSumType prefixSumType )
{
   /****
    * Compute the number of grids
    */
   const Index elementsInBlock = 8 * blockSize;
   const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
   const auto maxGridSize = Devices::Cuda::getMaxGridSize();
   const Index numberOfGrids = Devices::Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize );
   Array< DataType, Devices::Host, Index > gridShifts( numberOfGrids );
   gridShifts = zero;

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

      cudaGridPrefixSum( gridIdx == 0 ? prefixSumType : PrefixSumType::inclusive,
                         operation,
                         volatileOperation,
                         zero,
                         currentSize,
                         blockSize,
                         elementsInBlock,
                         &deviceInput[ gridOffset ],
                         &deviceOutput[ gridOffset ],
                         gridShifts[ gridIdx ] );
   }

   //gridShifts.computeExclusivePrefixSum();
   DataType aux( gridShifts[ 0 ] );
   gridShifts[ 0 ] = zero;
   for( Index i = 1; i < numberOfGrids; i++ )
   {
      DataType x = gridShifts[ i ];
      gridShifts[ i ] = aux;
      operation( aux, x );
   }

   for( Index gridIdx = 1; gridIdx < numberOfGrids; gridIdx ++ )
   {
      const Index gridOffset = gridIdx * maxGridSize * elementsInBlock;
      Index currentSize = size - gridOffset;
      if( currentSize / elementsInBlock > maxGridSize )
         currentSize = maxGridSize * elementsInBlock;
      //ArrayView< DataType, Devices::Cuda, Index > v( &deviceOutput[ gridOffset ], currentSize );
      const auto g = gridShifts[ gridIdx ];
      auto shift = [=] __cuda_callable__ ( Index i ) { deviceOutput[ gridOffset + i ] += g; };
      ParallelFor< Devices::Cuda >::exec( ( Index ) 0, currentSize, shift );
   }
}
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL


