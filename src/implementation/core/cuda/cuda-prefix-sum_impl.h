/***************************************************************************
                          cuda-prefix-sum_impl.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef CUDA_PREFIX_SUM_IMPL_H_
#define CUDA_PREFIX_SUM_IMPL_H_

template< typename DataType,
          template< typename T > class Operation,
          typename Index >
__global__ void cudaFirstPhaseBlockPrefixSum( const enumPrefixSumType prefixSumType,
                                              const Index size,
                                              const Index elementsInBlock,
                                              const DataType* input,
                                              DataType* output,
                                              DataType* auxArray )
{
   DataType* sharedData = sharedMemory< DataType >();
   DataType* auxData = &sharedData[ elementsInBlock + elementsInBlock / shmBanks + 2 ];
   DataType* warpSums = &auxData[ blockDim. x ];
   Operation< DataType > operation;

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
         sharedData[ interleave< shmBanks >( 0 ) ] = operation. cudaIdentity();
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ interleave< shmBanks >( idx + 1 ) ] = input[ blockOffset + idx ];
         idx += blockDim. x;
      }
   }
   else
      while( idx < elementsInBlock && blockOffset + idx < size )
      {
         sharedData[ interleave< shmBanks >( idx ) ] = input[ blockOffset + idx ];
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
         sharedData[ interleave< shmBanks >( chunkOffset ) ];
   }

   Index chunkPointer( 1 );
   while( chunkPointer < chunkSize &&
          chunkOffset + chunkPointer < lastElementInBlock )
   {
      operation. cudaPerformInPlace( sharedData[ interleave< shmBanks >( chunkOffset + chunkPointer ) ],
                                 sharedData[ interleave< shmBanks >( chunkOffset + chunkPointer - 1 ) ] );
      auxData[ threadIdx. x ] =
         sharedData[ interleave< shmBanks >( chunkOffset + chunkPointer  ) ];
      chunkPointer ++;
   }

   /***
    *  Perform the parallel prefix-sum inside warps.
    */
   const int threadInWarpIdx = threadIdx. x % warpSize;
   const int warpIdx = threadIdx. x / warpSize;
   for( int stride = 1; stride < warpSize; stride *= 2 )
      if( threadInWarpIdx >= stride && threadIdx. x < numberOfChunks )
         operation. cudaPerformInPlace( auxData[ threadIdx. x ], auxData[ threadIdx. x - stride ] );

   if( threadInWarpIdx == warpSize - 1 )
      warpSums[ warpIdx ] = auxData[ threadIdx. x ];
   __syncthreads();

   /****
    * Compute prefix-sum of warp sums using one warp
    */
   if( warpIdx == 0 )
      for( int stride = 1; stride < warpSize; stride *= 2 )
         if( threadInWarpIdx >= stride )
            operation. cudaPerformInPlace( warpSums[ threadInWarpIdx ], warpSums[ threadInWarpIdx - stride ] );
   __syncthreads();

   /****
    * Shift the warp prefix-sums.
    */
   if( warpIdx > 0 )
      operation. cudaPerformInPlace( auxData[ threadIdx. x ], warpSums[ warpIdx - 1 ] );

   /***
    *  Store the result back in global memory.
    */
   __syncthreads();
   idx = threadIdx. x;
   while( idx < elementsInBlock && blockOffset + idx < size )
   {
      const Index chunkIdx = idx / chunkSize;
      Index chunkShift( operation. cudaIdentity() );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      operation. cudaPerformInPlace( sharedData[ interleave< shmBanks >( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ interleave< shmBanks >( idx ) ];
      idx += blockDim. x;
   }
   __syncthreads();

   if( threadIdx. x == 0 )
   {
      if( prefixSumType == exclusivePrefixSum )
         auxArray[ blockIdx. x ] =
            operation. cudaPerform( sharedData[ interleave< shmBanks >( lastElementInBlock - 1 ) ],
                                    sharedData[ interleave< shmBanks >( lastElementInBlock ) ] );
      else
         auxArray[ blockIdx. x ] = sharedData[ interleave< shmBanks >( lastElementInBlock - 1 ) ];
   }

}

template< typename DataType,
          template< typename T > class Operation,
          typename Index >
__global__ void cudaSecondPhaseBlockPrefixSum( const Index size,
                                               const Index elementsInBlock,
                                               const Index gridShift,
                                               const DataType* auxArray,
                                               DataType* data )
{
   Operation< DataType > operation;
   if( blockIdx. x > 0 )
   {
      const Index shift = operation. cudaPerform( gridShift, auxArray[ blockIdx. x - 1 ] );

      const Index readOffset = blockIdx. x * elementsInBlock;
      Index readIdx = threadIdx. x;
      while( readIdx < elementsInBlock && readOffset + readIdx < size )
      {
         operation. cudaPerformInPlace( data[ readIdx + readOffset ], shift );
         readIdx += blockDim. x;
      }
   }
}


template< typename DataType,
          template< typename T > class Operation,
          typename Index >
bool cudaRecursivePrefixSum( const enumPrefixSumType prefixSumType,
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
         cerr << "Not enough memory on device to allocate auxilliary arrays." << endl;
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
   size_t sharedDataSize = elementsInBlock + elementsInBlock / shmBanks + 2;
   size_t sharedMemory = ( sharedDataSize + blockSize + warpSize  ) * sizeof( DataType );
   cudaFirstPhaseBlockPrefixSum< DataType, Operation, Index >
                                <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                (  prefixSumType,
                                   size,
                                   elementsInBlock,
                                   input,
                                   output,
                                   auxArray1 );
   cudaError error = cudaGetLastError();
   if( error != cudaSuccess )
   {
      cerr << "The CUDA kernel 'cudaFirstPhaseBlockPrefixSum' ended with error code " << error << "."
           << endl;
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
                                 numberOfBlocks,
                                 blockSize,
                                 elementsInBlock,
                                 0,
                                 auxArray1,
                                 auxArray2 ) )
      return false;
   cudaSecondPhaseBlockPrefixSum< DataType, Operation, Index >
                                <<< cudaGridSize, cudaBlockSize >>>
                                 ( size, elementsInBlock, gridShift, auxArray2, output );
   error = cudaGetLastError();
   if( error != cudaSuccess )
   {
      cerr << "The CUDA kernel 'cudaSecondPhaseBlockPrefixSum' ended with error code " << error << "."
           << endl;
      return false;
   }
   cudaFree( auxArray1 );
   cudaFree( auxArray2 );
   return true;
}



template< typename DataType,
          template< typename T > class Operation,
          typename Index >
bool cudaGridPrefixSum( enumPrefixSumType prefixSumType,
                        const Index size,
                        const Index blockSize,
                        const Index elementsInBlock,
                        const DataType *deviceInput,
                        DataType *deviceOutput,
                        Index& gridShift )
{

   if( ! cudaRecursivePrefixSum< DataType, Operation, Index >
                               ( prefixSumType,
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
      cerr << "I am not able to copy data from device to host." << endl;
      return false;
   }
   return true;
}

template< typename DataType,
          template< typename T > class Operation,
          typename Index >
bool cudaPrefixSum( const Index size,
                    const Index blockSize,
                    const DataType *deviceInput,
                    DataType* deviceOutput,
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

#endif /* CUDA_PREFIX_SUM_IMPL_H_ */
