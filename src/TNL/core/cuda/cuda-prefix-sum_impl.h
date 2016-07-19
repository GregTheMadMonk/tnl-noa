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
#include <TNL/core/tnlCuda.h>
#include <TNL/core/cuda/reduction-operations.h>
   
#ifdef HAVE_CUDA

namespace TNL {

template< typename DataType,
          typename Operation,
          typename Index >
__global__ void cudaFirstPhaseBlockPrefixSum( const enumPrefixSumType prefixSumType,
                                              Operation operation,
                                              const Index size,
                                              const Index elementsInBlock,
                                              const DataType* input,
                                              DataType* output,
                                              DataType* auxArray )
{
   DataType* sharedData = getSharedMemory< DataType >();
   DataType* auxData = &sharedData[ elementsInBlock + elementsInBlock / tnlCuda::getNumberOfSharedMemoryBanks() + 2 ];
   DataType* warpSums = &auxData[ blockDim. x ];

   const Index lastElementIdx = size - blockIdx. x * elementsInBlock;
   Index lastElementInBlock = ( lastElementIdx < elementsInBlock ?
                                lastElementIdx : elementsInBlock );

   /***
    * Load data into the shared memory.
    */
   const Index blockOffset = blockIdx. x * elementsInBlock;
   Index idx = threadIdx. x;
   if( prefixSumType == exclusivePrefixSum )
   {
      if( idx == 0 )
         sharedData[ 0 ] = operation.initialValue();
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ tnlCuda::getInterleaving( idx + 1 ) ] = input[ blockOffset + idx ];
         idx += blockDim. x;
      }
   }
   else
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ tnlCuda::getInterleaving( idx ) ] = input[ blockOffset + idx ];
         idx += blockDim. x;
      }

   /***
    * Perform the sequential prefix-sum.
    */
   __syncthreads();
   const int chunkSize = elementsInBlock / blockDim. x;
   const int chunkOffset = threadIdx. x * chunkSize;
   const int numberOfChunks = lastElementInBlock / chunkSize +
                            ( lastElementInBlock % chunkSize != 0 );

   if( chunkOffset < lastElementInBlock )
   {
      auxData[ threadIdx. x ] =
         sharedData[ tnlCuda::getInterleaving( chunkOffset ) ];
   }

   Index chunkPointer( 1 );
   while( chunkPointer < chunkSize &&
          chunkOffset + chunkPointer < lastElementInBlock )
   {
      operation.commonReductionOnDevice( sharedData[ tnlCuda::getInterleaving( chunkOffset + chunkPointer ) ],
                                         sharedData[ tnlCuda::getInterleaving( chunkOffset + chunkPointer - 1 ) ] );
      auxData[ threadIdx. x ] =
         sharedData[ tnlCuda::getInterleaving( chunkOffset + chunkPointer  ) ];
      chunkPointer ++;
   }

   /***
    *  Perform the parallel prefix-sum inside warps.
    */
   const int threadInWarpIdx = threadIdx. x % tnlCuda::getWarpSize();
   const int warpIdx = threadIdx. x / tnlCuda::getWarpSize();
   for( int stride = 1; stride < tnlCuda::getWarpSize(); stride *= 2 )
      if( threadInWarpIdx >= stride && threadIdx. x < numberOfChunks )
         operation.commonReductionOnDevice( auxData[ threadIdx. x ], auxData[ threadIdx. x - stride ] );

   if( threadInWarpIdx == tnlCuda::getWarpSize() - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx. x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < tnlCuda::getWarpSize(); stride *= 2 )
         if( threadInWarpIdx >= stride )
            operation.commonReductionOnDevice( warpSums[ threadInWarpIdx ], warpSums[ threadInWarpIdx - stride ] );
   __syncthreads();

   /****
    * Shift the warp prefix-sums.
    */
   if( warpIdx > 0 )
      operation.commonReductionOnDevice( auxData[ threadIdx. x ], warpSums[ warpIdx - 1 ] );

   /***
    *  Store the result back in global memory.
    */
   __syncthreads();
   idx = threadIdx. x;
   while( idx < elementsInBlock && blockOffset + idx < size )
   {
      const Index chunkIdx = idx / chunkSize;
      DataType chunkShift( operation.initialValue() );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      operation.commonReductionOnDevice( sharedData[ tnlCuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ tnlCuda::getInterleaving( idx ) ];
      idx += blockDim. x;
   }
   __syncthreads();

   if( threadIdx. x == 0 )
   {
      if( prefixSumType == exclusivePrefixSum )
      {
         /*auxArray[ blockIdx. x ] = operation.commonReductionOnDevice( tnlCuda::getInterleaving( lastElementInBlock - 1 ),
                                                                      tnlCuda::getInterleaving( lastElementInBlock ),
                                                                      sharedData );*/
         DataType aux = operation.initialValue();
         operation.commonReductionOnDevice( aux, sharedData[ tnlCuda::getInterleaving( lastElementInBlock - 1 ) ] );
         operation.commonReductionOnDevice( aux, sharedData[ tnlCuda::getInterleaving( lastElementInBlock ) ] );
         auxArray[ blockIdx. x ] = aux;
      }
      else
         auxArray[ blockIdx. x ] = sharedData[ tnlCuda::getInterleaving( lastElementInBlock - 1 ) ];
   }

}

template< typename DataType,
          typename Operation,
          typename Index >
__global__ void cudaSecondPhaseBlockPrefixSum( Operation operation,
                                               const Index size,
                                               const Index elementsInBlock,
                                               const Index gridShift,
                                               const DataType* auxArray,
                                               DataType* data )
{
   if( blockIdx. x > 0 )
   {
      DataType shift( gridShift );
      operation.commonReductionOnDevice( shift, auxArray[ blockIdx. x - 1 ] );

      const Index readOffset = blockIdx. x * elementsInBlock;
      Index readIdx = threadIdx. x;
      while( readIdx < elementsInBlock && readOffset + readIdx < size )
      {
         operation.commonReductionOnDevice( data[ readIdx + readOffset ], shift );
         readIdx += blockDim. x;
      }
   }
}


template< typename DataType,
          typename Operation,
          typename Index >
bool cudaRecursivePrefixSum( const enumPrefixSumType prefixSumType,
                             Operation& operation,
                             const Index size,
                             const Index blockSize,
                             const Index elementsInBlock,
                             const Index gridShift,
                             const DataType* input,
                             DataType *output )
{
   const Index numberOfBlocks = ceil( ( double ) size / ( double ) elementsInBlock );
   const Index auxArraySize = numberOfBlocks * sizeof( DataType );
   DataType *auxArray1, *auxArray2;

   if( cudaMalloc( ( void** ) &auxArray1, auxArraySize ) != cudaSuccess ||
       cudaMalloc( ( void** ) &auxArray2, auxArraySize ) != cudaSuccess  )
   {
      {
         std::cerr << "Not enough memory on device to allocate auxilliary arrays." << std::endl;
         return false;
      }
   }

   /****
    * Setup block and grid size.
    */
   dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
   cudaBlockSize. x = blockSize;
   cudaGridSize. x = size / elementsInBlock +
                     ( size % elementsInBlock != 0 );

   /****
    * Run the kernel.
    */
   size_t sharedDataSize = elementsInBlock +
                           elementsInBlock / tnlCuda::getNumberOfSharedMemoryBanks() + 2;
   size_t sharedMemory = ( sharedDataSize + blockSize + tnlCuda::getWarpSize()  ) * sizeof( DataType );
   cudaFirstPhaseBlockPrefixSum< DataType, Operation, Index >
                                <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                (  prefixSumType,
                                   operation,
                                   size,
                                   elementsInBlock,
                                   input,
                                   output,
                                   auxArray1 );
   if( ! checkCudaDevice )
   {
      std::cerr << "The CUDA kernel 'cudaFirstPhaseBlockPrefixSum' ended with error." << std::endl;
      cudaFree( auxArray1 );
      cudaFree( auxArray2 );
      return false;
   }

   /***
    * In auxArray1 there is now a sum of numbers in each block.
    * We must compute prefix-sum of auxArray1 and then shift
    * each block.
    */
   if( numberOfBlocks > 1 &&
       ! cudaRecursivePrefixSum< DataType, Operation, Index >
                               ( inclusivePrefixSum,
                                 operation,
                                 numberOfBlocks,
                                 blockSize,
                                 elementsInBlock,
                                 0,
                                 auxArray1,
                                 auxArray2 ) )
      return false;
   cudaSecondPhaseBlockPrefixSum< DataType, Operation, Index >
                                <<< cudaGridSize, cudaBlockSize >>>
                                 ( operation, size, elementsInBlock, gridShift, auxArray2, output );

   if( ! checkCudaDevice )
   {
      std::cerr << "The CUDA kernel 'cudaSecondPhaseBlockPrefixSum' ended with error." << std::endl;
      cudaFree( auxArray1 );
      cudaFree( auxArray2 );
      return false;
   }
   cudaFree( auxArray1 );
   cudaFree( auxArray2 );
   return true;
}



template< typename DataType,
          typename Operation,
          typename Index >
bool cudaGridPrefixSum( enumPrefixSumType prefixSumType,
                        Operation& operation,
                        const Index size,
                        const Index blockSize,
                        const Index elementsInBlock,
                        const DataType *deviceInput,
                        DataType *deviceOutput,
                        Index& gridShift )
{

   if( ! cudaRecursivePrefixSum< DataType, Operation, Index >
                               ( prefixSumType,
                                 operation,
                                 size,
                                 blockSize,
                                 elementsInBlock,
                                 gridShift,
                                 deviceInput,
                                 deviceOutput ) )
      return false;
   if( cudaMemcpy( &gridShift,
                   &deviceOutput[ size - 1 ],
                   sizeof( DataType ),
                   cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      std::cerr << "I am not able to copy data from device to host." << std::endl;
      return false;
   }
   return true;
}

template< typename DataType,
          typename Operation,
          typename Index >
bool cudaPrefixSum( const Index size,
                    const Index blockSize,
                    const DataType *deviceInput,
                    DataType* deviceOutput,
                    Operation& operation,
                    const enumPrefixSumType prefixSumType )
{
   /****
    * Compute the number of grids
    */
   const Index elementsInBlock = 8 * blockSize;
   const Index gridSize = size / elementsInBlock + ( size % elementsInBlock != 0 );
   const Index maxGridSize = 65536;
   const Index gridsNumber = gridSize / maxGridSize + ( gridSize % maxGridSize != 0 );

   /****
    * Loop over all grids.
    */
   Index gridShift( 0 );
   for( Index gridIdx = 0; gridIdx < gridsNumber; gridIdx ++ )
   {
      /****
       * Compute current grid size and size of data to be scanned
       */
      Index gridSize = ( size - gridIdx * maxGridSize * elementsInBlock ) /
                     elementsInBlock;
      Index currentSize = size - gridIdx * maxGridSize * elementsInBlock;
      if( gridSize > maxGridSize )
      {
         gridSize = maxGridSize;
         currentSize = maxGridSize * elementsInBlock;
      }
      Index gridOffset = gridIdx * maxGridSize * elementsInBlock;
      if( ! cudaGridPrefixSum< DataType, Operation, Index >
                             ( prefixSumType,
                               operation,
                               currentSize,
                               blockSize,
                               elementsInBlock,
                               &deviceInput[ gridOffset ],
                               &deviceOutput[ gridOffset ],
                               gridShift ) )
         return false;
   }
   return true;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const int *deviceInput,
                                    int* deviceOutput,
                                    tnlParallelReductionSum< int, int >& operation,
                                    const enumPrefixSumType prefixSumType );


extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const float *deviceInput,
                                    float* deviceOutput,
                                    tnlParallelReductionSum< float, int >& operation,
                                    const enumPrefixSumType prefixSumType );

extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const double *deviceInput,
                                    double* deviceOutput,
                                    tnlParallelReductionSum< double, int >& operation,
                                    const enumPrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool cudaPrefixSum( const int size,
                                    const int blockSize,
                                    const long double *deviceInput,
                                    long double* deviceOutput,
                                    tnlParallelReductionSum< long double, int >& operation,
                                    const enumPrefixSumType prefixSumType );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const int *deviceInput,
                                    int* deviceOutput,
                                    tnlParallelReductionSum< int, long int >& operation,
                                    const enumPrefixSumType prefixSumType );


extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const float *deviceInput,
                                    float* deviceOutput,
                                    tnlParallelReductionSum< float, long int >& operation,
                                    const enumPrefixSumType prefixSumType );

extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const double *deviceInput,
                                    double* deviceOutput,
                                    tnlParallelReductionSum< double, long int >& operation,
                                    const enumPrefixSumType prefixSumType );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool cudaPrefixSum( const long int size,
                                    const long int blockSize,
                                    const long double *deviceInput,
                                    long double* deviceOutput,
                                    tnlParallelReductionSum< long double, long int >& operation,
                                    const enumPrefixSumType prefixSumType );
#endif
#endif

#endif

} // namespace TNL

#endif
