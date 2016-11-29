/***************************************************************************
                          CudaReductionKernel.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Containers/Algorithms/CudaReductionBuffer.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef HAVE_CUDA
/****
 * The performance of this kernel is very sensitive to register usage.
 * Compile with --ptxas-options=-v and configure these constants for given
 * architecture so that there are no local memory spills.
 */
static constexpr int Reduction_maxThreadsPerBlock = 256;  // must be a power of 2
#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Reduction_minBlocksPerMultiprocessor = 8;
#else
   static constexpr int Reduction_minBlocksPerMultiprocessor = 4;
#endif

template< typename Operation, int blockSize >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionKernel( Operation operation,
                     const typename Operation::IndexType size,
                     const typename Operation::RealType* input1,
                     const typename Operation::RealType* input2,
                     typename Operation::ResultType* output )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::ResultType ResultType;

   ResultType* sdata = Devices::Cuda::getSharedMemory< ResultType >();

   /***
    * Get thread id (tid) and global thread id (gid).
    * gridSize is the number of element processed by all blocks at the
    * same time.
    */
   const IndexType tid = threadIdx.x;
         IndexType gid = blockIdx.x * blockDim. x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   sdata[ tid ] = operation.initialValue();
   /***
    * Read data into the shared memory. We start with the
    * sequential reduction.
    */
   while( gid + 4 * gridSize < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + gridSize,     input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + 2 * gridSize, input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + 3 * gridSize, input1, input2 );
      gid += 4 * gridSize;
   }
   while( gid + 2 * gridSize < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + gridSize,     input1, input2 );
      gid += 2 * gridSize;
   }
   while( gid < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                input1, input2 );
      gid += gridSize;
   }
   __syncthreads();


   //printf( "1: tid %d data %f \n", tid, sdata[ tid ] );

   //return;
   /***
    *  Perform the parallel reduction.
    */
   if( blockSize >= 1024 )
   {
      if( tid < 512 )
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSize >= 512 )
   {
      if( tid < 256 )
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSize >= 256 )
   {
      if( tid < 128 )
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
      //printf( "2: tid %d data %f \n", tid, sdata[ tid ] );
   }

   if( blockSize >= 128 )
   {
      if( tid <  64 )
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
      //printf( "3: tid %d data %f \n", tid, sdata[ tid ] );
   }


   /***
    * This runs in one warp so it is synchronized implicitly.
    */
   if( tid < 32 )
   {
      volatile ResultType* vsdata = sdata;
      if( blockSize >= 64 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 32 ] );
         //printf( "4: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >= 32 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 16 ] );
         //printf( "5: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >= 16 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 8 ] );
         //printf( "6: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  8 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 4 ] );
         //printf( "7: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  4 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 2 ] );
         //printf( "8: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  2 )
      {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 1 ] );
         //printf( "9: tid %d data %f \n", tid, sdata[ tid ] );
      }
   }

   /***
    * Store the result back in the global memory.
    */
   if( tid == 0 )
   {
      //printf( "Block %d result = %f \n", blockIdx.x, sdata[ 0 ] );
      output[ blockIdx.x ] = sdata[ 0 ];
   }

}

template< typename Operation >
typename Operation::IndexType
CudaReductionKernelLauncher( Operation& operation,
                             const typename Operation::IndexType size,
                             const typename Operation::RealType* input1,
                             const typename Operation::RealType* input2,
                             typename Operation::ResultType*& output )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;
   typedef typename Operation::ResultType ResultType;

   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // On Tesla K40c, desGridSize = 4 * 6 * 15 = 360.
//   const IndexType desGridSize = 4 * Reduction_minBlocksPerMultiprocessor
//                                   * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   // On Tesla K40c, desGridSize = 6 * 15 = 90.
   const IndexType desGridSize = Reduction_minBlocksPerMultiprocessor
                               * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   dim3 blockSize( 256 ), gridSize( 0 );
   gridSize.x = min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

   // create reference to the reduction buffer singleton and set size
   const size_t buf_size = desGridSize * sizeof( ResultType );
   CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
   if( ! cudaReductionBuffer.setSize( buf_size ) )
      throw 1;
   output = cudaReductionBuffer.template getData< ResultType >();

   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   const IndexType shmem = (blockSize.x <= 32)
            ? 2 * blockSize.x * sizeof( ResultType )
            : blockSize.x * sizeof( ResultType );

   /***
    * Depending on the blockSize we generate appropriate template instance.
    */
   switch( blockSize.x )
   {
      case 512:
         CudaReductionKernel< Operation, 512 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case 256:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation, 256 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation, 256 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case 128:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation, 128 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation, 128 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  64:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,  64 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,  64 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  32:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,  32 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,  32 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  16:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,  16 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,  16 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
     case   8:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,   8 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,   8 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case   4:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,   4 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,   4 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case   2:
         cudaFuncSetCacheConfig(CudaReductionKernel< Operation,   2 >, cudaFuncCachePreferShared);

         CudaReductionKernel< Operation,   2 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case   1:
         TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
      default:
         TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
   }
   checkCudaDevice;

   // return the size of the output array on the CUDA device
   return gridSize.x;
}
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
