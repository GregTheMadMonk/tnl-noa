/***************************************************************************
                          cuda-prefix-sum_impl.h  -  description
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

#ifdef HAVE_CUDA

namespace TNL {
namespace Containers {
namespace Algorithms {   

template< typename DataType,
          typename Operation,
          typename Index >
__global__ void
cudaFirstPhaseBlockPrefixSum( const PrefixSumType prefixSumType,
                              Operation operation,
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
         sharedData[ 0 ] = operation.initialValue();
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
      operation.commonReduction( sharedData[ Devices::Cuda::getInterleaving( chunkOffset + chunkPointer ) ],
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
         operation.commonReduction( auxData[ threadIdx.x ], auxData[ threadIdx.x - stride ] );

   if( threadInWarpIdx == Devices::Cuda::getWarpSize() - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx.x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < Devices::Cuda::getWarpSize(); stride *= 2 )
         if( threadInWarpIdx >= stride )
            operation.commonReduction( warpSums[ threadIdx.x ], warpSums[ threadIdx.x - stride ] );
   __syncthreads();

   /****
    * Shift the warp prefix-sums.
    */
   if( warpIdx > 0 )
      operation.commonReduction( auxData[ threadIdx.x ], warpSums[ warpIdx - 1 ] );

   /***
    *  Store the result back in global memory.
    */
   __syncthreads();
   idx = threadIdx.x;
   while( idx < elementsInBlock && blockOffset + idx < size )
   {
      const Index chunkIdx = idx / chunkSize;
      DataType chunkShift( operation.initialValue() );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      operation.commonReduction( sharedData[ Devices::Cuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ Devices::Cuda::getInterleaving( idx ) ];
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( prefixSumType == PrefixSumType::exclusive )
      {
         /*auxArray[ blockIdx.x ] = operation.commonReduction( Devices::Cuda::getInterleaving( lastElementInBlock - 1 ),
                                                               Devices::Cuda::getInterleaving( lastElementInBlock ),
                                                               sharedData );*/
         DataType aux = operation.initialValue();
         operation.commonReduction( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ] );
         operation.commonReduction( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock ) ] );
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
      operation.commonReduction( gridShift, auxArray[ blockIdx.x - 1 ] );

      const Index readOffset = blockIdx.x * elementsInBlock;
      Index readIdx = threadIdx.x;
      while( readIdx < elementsInBlock && readOffset + readIdx < size )
      {
         operation.commonReduction( data[ readIdx + readOffset ], gridShift );
         readIdx += blockDim.x;
      }
   }
}


template< typename DataType,
          typename Operation,
          typename Index >
void
cudaRecursivePrefixSum( const PrefixSumType prefixSumType,
                        Operation& operation,
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
                               numberOfBlocks,
                               blockSize,
                               elementsInBlock,
                               operation.initialValue(),
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
          typename Index >
void
cudaGridPrefixSum( PrefixSumType prefixSumType,
                   Operation& operation,
                   const Index size,
                   const Index blockSize,
                   const Index elementsInBlock,
                   const DataType *deviceInput,
                   DataType *deviceOutput,
                   DataType& gridShift )
{
   cudaRecursivePrefixSum( prefixSumType,
                           operation,
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

template< typename DataType,
          typename Operation,
          typename Index >
void
cudaPrefixSum( const Index size,
               const Index blockSize,
               const DataType *deviceInput,
               DataType* deviceOutput,
               Operation& operation,
               const PrefixSumType prefixSumType )
{
   /****
    * Compute the number of grids
    */
   const Index elementsInBlock = 8 * blockSize;
   const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
   const auto maxGridSize = Devices::Cuda::getMaxGridSize();
   const Index numberOfGrids = Devices::Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize );

   /****
    * Loop over all grids.
    */
   DataType gridShift = operation.initialValue();
   for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ )
   {
      /****
       * Compute current grid size and size of data to be scanned
       */
      const Index gridOffset = gridIdx * maxGridSize * elementsInBlock;
      Index currentSize = size - gridOffset;
      if( currentSize / elementsInBlock > maxGridSize )
         currentSize = maxGridSize * elementsInBlock;

      cudaGridPrefixSum( prefixSumType,
                         operation,
                         currentSize,
                         blockSize,
                         elementsInBlock,
                         &deviceInput[ gridOffset ],
                         &deviceOutput[ gridOffset ],
                         gridShift );
   }
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const int *deviceInput,
                                    int* deviceOutput,
                                    tnlParallelReductionSum< int, int >& operation,
                                    const PrefixSumType prefixSumType );


extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const float *deviceInput,
                                    float* deviceOutput,
                                    tnlParallelReductionSum< float, int >& operation,
                                    const PrefixSumType prefixSumType );

extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const double *deviceInput,
                                    double* deviceOutput,
                                    tnlParallelReductionSum< double, int >& operation,
                                    const PrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const long double *deviceInput,
                                    long double* deviceOutput,
                                    tnlParallelReductionSum< long double, int >& operation,
                                    const PrefixSumType prefixSumType );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const int *deviceInput,
                                    int* deviceOutput,
                                    tnlParallelReductionSum< int, long int >& operation,
                                    const PrefixSumType prefixSumType );


extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const float *deviceInput,
                                    float* deviceOutput,
                                    tnlParallelReductionSum< float, long int >& operation,
                                    const PrefixSumType prefixSumType );

extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const double *deviceInput,
                                    double* deviceOutput,
                                    tnlParallelReductionSum< double, long int >& operation,
                                    const PrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const long double *deviceInput,
                                    long double* deviceOutput,
                                    tnlParallelReductionSum< long double, long int >& operation,
                                    const PrefixSumType prefixSumType );
#endif
#endif

#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#endif
