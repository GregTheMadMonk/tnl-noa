/***************************************************************************
                          CudaMultireductionKernel.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

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
static constexpr int Multireduction_maxThreadsPerBlock = 256;  // must be a power of 2
#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 6;
#else
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 4;
#endif

template< typename Operation, int blockSizeX >      
__global__ void
__launch_bounds__( Multireduction_maxThreadsPerBlock, Multireduction_minBlocksPerMultiprocessor )
CudaMultireductionKernel( Operation operation,
                          const typename Operation::IndexType n,
                          const typename Operation::IndexType size,
                          const typename Operation::RealType* input1,
                          const typename Operation::IndexType ldInput1,
                          const typename Operation::RealType* input2,
                          typename Operation::ResultType* output )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::ResultType ResultType;

   ResultType* sdata = Devices::Cuda::getSharedMemory< ResultType >();

   /***
    * Get thread id (tid) and global element id (gid).
    * gridSizeX is the number of elements in the direction of x-axis
    * processed by all blocks at the same time.
    */
   const IndexType tid = threadIdx.y * blockDim.x + threadIdx.x;
         IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSizeX = blockDim.x * gridDim.x;

   /***
    * Shift input1 and output pointers.
    */
   const IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
   if( y < n ) {
      input1 += y * ldInput1;
      output += y * gridDim.x;
   }
   else
      return;

   /***
    * Start with the sequential reduction and push the
    * result into the shared memory.
    */
   sdata[ tid ] = operation.initialValue();
   while( gid + 4 * gridSizeX < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                 input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + gridSizeX,     input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + 2 * gridSizeX, input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + 3 * gridSizeX, input1, input2 );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                 input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + gridSizeX,     input1, input2 );
      gid += 2 * gridSizeX;
   }
   while( gid < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                 input1, input2 );
      gid += gridSizeX;
   }
   __syncthreads();


   //printf( "1: tid %d data %f \n", tid, sdata[ tid ] );

   /***
    *  Perform the parallel reduction.
    */
   if( blockSizeX >= 1024 ) {
      if( threadIdx.x < 512 ) {
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 512 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 512 ) {
      if( threadIdx.x < 256 ) {
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 256 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 256 ) {
      if( threadIdx.x < 128 ) {
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 128 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 128 ) {
      if( threadIdx.x <  64 ) {
         operation.commonReductionOnDevice( sdata[ tid ], sdata[ tid + 64 ] );
      }
      __syncthreads();
   }

   /***
    * This runs in one warp so it is synchronized implicitly.
    *
    * When the blockSizeX is less then or equal to the warp size, the shared memory
    * must be at least 2 * blockSizeX elements per block, otherwise unallocated memory
    * will be accessed !!!
    */
   if( threadIdx.x < 32 ) {
      volatile ResultType* vsdata = sdata;
      if( blockSizeX >= 64 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 32 ] );
      }
      if( blockSizeX >= 32 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 16 ] );
      }
      if( blockSizeX >= 16 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 8 ] );
      }
      if( blockSizeX >=  8 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 4 ] );
      }
      if( blockSizeX >=  4 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 2 ] );
      }
      if( blockSizeX >=  2 ) {
         operation.commonReductionOnDevice( vsdata[ tid ], vsdata[ tid + 1 ] );
      }
   }

   /***
    * Store the result back in the global memory.
    */
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x ] = sdata[ tid ];
   }
}

template< typename Operation >
typename Operation::IndexType
CudaMultireductionKernelLauncher( Operation& operation,
                                  int n,
                                  const typename Operation::IndexType size,
                                  const typename Operation::RealType* input1,
                                  const typename Operation::IndexType ldInput1,
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
   // On Tesla K40c, desGridSizeX = 4 * 6 * 15 = 360.
//   const IndexType desGridSizeX = 4 * Multireduction_minBlocksPerMultiprocessor
//                                    * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   // On Tesla K40c, desGridSizeX = 6 * 15 = 90.
   const IndexType desGridSizeX = Multireduction_minBlocksPerMultiprocessor
                                * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   dim3 blockSize, gridSize;
   
   // version A: max 16 rows of threads
   blockSize.y = min( n, 16 );

   // version B: up to 16 rows of threads, then "minimize" number of inactive rows
//   if( n <= 16 )
//      blockSize.y = n;
//   else {
//      int r = (n - 1) % 16 + 1;
//      if( r > 12 )
//         blockSize.y = 16;
//      else if( r > 8 )
//         blockSize.y = 4;
//      else if( r > 4 )
//         blockSize.y = 8;
//      else
//         blockSize.y = 4;
//   }

   // blockSize.x has to be a power of 2
   blockSize.x = Multireduction_maxThreadsPerBlock;
   while( blockSize.x * blockSize.y > Multireduction_maxThreadsPerBlock )
      blockSize.x /= 2;

   gridSize.x = min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSizeX );
   gridSize.y = Devices::Cuda::getNumberOfBlocks( n, blockSize.y );

   if( gridSize.y > (unsigned) Devices::Cuda::getMaxGridSize() ) {
      std::cerr << "Maximum gridSize.y limit exceeded (limit is 65535, attempted " << gridSize.y << ")." << std::endl;
      throw 1;
   }

   // create reference to the reduction buffer singleton and set size
   // (make an overestimate to avoid reallocation on every call if n is increased by 1 each time)
   const size_t buf_size = 8 * ( n / 8 + 1 ) * desGridSizeX * sizeof( ResultType );
   CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
   if( ! cudaReductionBuffer.setSize( buf_size ) )
      throw 1;
   output = cudaReductionBuffer.template getData< ResultType >();

   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   const IndexType shmem = (blockSize.x <= 32)
            ? 2 * blockSize.x * blockSize.y * sizeof( ResultType )
            : blockSize.x * blockSize.y * sizeof( ResultType );

   //cout << "Multireduction of " << n << " datasets, block size (" << blockSize.x << "," << blockSize.y << "), grid size (" << gridSize.x << "," << gridSize.y << "), shmem " << shmem << endl;

   /***
    * Depending on the blockSize we generate appropriate template instance.
    */
   switch( blockSize.x )
   {
      case 512:
         CudaMultireductionKernel< Operation, 512 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case 256:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation, 256 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation, 256 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case 128:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation, 128 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation, 128 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  64:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,  64 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,  64 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  32:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,  32 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,  32 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  16:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,  16 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,  16 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
     case   8:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,   8 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,   8 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case   4:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,   4 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,   4 >
        <<< gridSize, blockSize, shmem >>>( operation,  n,size, input1, ldInput1, input2, output);
        break;
      case   2:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< Operation,   2 >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< Operation,   2 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case   1:
         TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
      default:
         TNL_ASSERT( false, std::cerr << "Block size is " << blockSize.x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." << std::endl );
   }
   checkCudaDevice;

   // return the size of the output array on the CUDA device
   return gridSize.x;
}
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
