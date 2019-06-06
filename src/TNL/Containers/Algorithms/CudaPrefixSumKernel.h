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
#include <TNL/Containers/Algorithms/ReductionOperations.h>
#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef HAVE_CUDA

template< typename Real,
          typename Operation,
          typename VolatileOperation,
          typename Index >
__global__ void
cudaFirstPhaseBlockPrefixSum( const PrefixSumType prefixSumType,
                              Operation operation,
                              VolatileOperation volatileOperation,
                              const Real zero,
                              const Index size,
                              const Index elementsInBlock,
                              const Real* input,
                              Real* output,
                              Real* auxArray,
                              const Real gridShift )
{
   Real* sharedData = TNL::Devices::Cuda::getSharedMemory< Real >();
   volatile Real* auxData = &sharedData[ elementsInBlock + elementsInBlock / Devices::Cuda::getNumberOfSharedMemoryBanks() + 2 ];
   volatile Real* warpSums = &auxData[ blockDim.x ];

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
      operation( sharedData[ 0 ], gridShift );

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
      Real chunkShift( zero );
      if( chunkIdx > 0 )
         chunkShift = auxData[ chunkIdx - 1 ];
      operation( sharedData[ Devices::Cuda::getInterleaving( idx ) ], chunkShift );
      output[ blockOffset + idx ] = sharedData[ Devices::Cuda::getInterleaving( idx ) ];
      idx += blockDim.x;
   }
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      if( prefixSumType == PrefixSumType::Exclusive )
      {
         Real aux = zero;
         operation( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ] );
         operation( aux, sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock ) ] );
         auxArray[ blockIdx.x ] = aux;
      }
      else
         auxArray[ blockIdx.x ] = sharedData[ Devices::Cuda::getInterleaving( lastElementInBlock - 1 ) ];
   }
}

template< typename Real,
          typename Operation,
          typename Index >
__global__ void
cudaSecondPhaseBlockPrefixSum( Operation operation,
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
         operation( data[ readIdx + readOffset ], shift );
         readIdx += blockDim.x;
      }
   }
}

template< PrefixSumType prefixSumType,
          typename Real,
          typename Index >
struct CudaPrefixSumKernelLauncher
{
   template< typename Operation,
             typename VolatileOperation >
   static void
   cudaRecursivePrefixSum( PrefixSumType prefixSumType_,
                           Operation& operation,
                           VolatileOperation& volatileOperation,
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
      const std::size_t sharedMemory = ( sharedDataSize + blockSize + Devices::Cuda::getWarpSize()  ) * sizeof( Real );
      cudaFirstPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize, sharedMemory >>>
         ( prefixSumType_,
           operation,
           volatileOperation,
           zero,
           size,
           elementsInBlock,
           input,
           output,
           auxArray1.getData(),
           gridShift );
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
            operation,
            volatileOperation,
            zero,
            numberOfBlocks,
            blockSize,
            elementsInBlock,
            gridShift2,
            auxArray1.getData(),
            auxArray2.getData() );

      //std::cerr << " auxArray2 = " << auxArray2 << std::endl;
      cudaSecondPhaseBlockPrefixSum<<< cudaGridSize, cudaBlockSize >>>
         ( operation,
           size,
           elementsInBlock,
           gridShift,
           auxArray2.getData(),
           output );
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
    * \tparam Operation operation to be performed on particular elements - addition usually
    * \tparam VolatileOperation - volatile version of Operation
    * \param size is number of elements to be scanned
    * \param blockSize is CUDA block size
    * \param deviceInput is pointer to input data on GPU
    * \param deviceOutput is pointer to resulting array, can be the same as input
    * \param operation is instance of Operation
    * \param volatileOperation is instance of VolatileOperation
    * \param zero is neutral element for given Operation
    */
   template< typename Operation,
             typename VolatileOperation >
   static void
   start( const Index size,
      const Index blockSize,
      const Real *deviceInput,
      Real* deviceOutput,
      Operation& operation,
      VolatileOperation& volatileOperation,
      const Real& zero )
   {
      /****
       * Compute the number of grids
       */
      const Index elementsInBlock = 8 * blockSize;
      const Index numberOfBlocks = roundUpDivision( size, elementsInBlock );
      const auto maxGridSize = 3; //Devices::Cuda::getMaxGridSize();
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
            operation,
            volatileOperation,
            zero,
            currentSize,
            blockSize,
            elementsInBlock,
            gridShift,
            &deviceInput[ gridOffset ],
            &deviceOutput[ gridOffset ] );
         TNL_CHECK_CUDA_DEVICE;
      }
   }
};
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL


