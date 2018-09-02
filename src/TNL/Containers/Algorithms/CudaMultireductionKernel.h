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
static constexpr int Multireduction_registersPerThread = 38;   // empirically determined optimal value

// __CUDA_ARCH__ is defined only in device code!
#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 6;
#else
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 4;
#endif

template< int blockSizeX, typename Operation, typename Index >
__global__ void
__launch_bounds__( Multireduction_maxThreadsPerBlock, Multireduction_minBlocksPerMultiprocessor )
CudaMultireductionKernel( Operation operation,
                          const int n,
                          const Index size,
                          const typename Operation::DataType1* input1,
                          const Index ldInput1,
                          const typename Operation::DataType2* input2,
                          typename Operation::ResultType* output )
{
   typedef Index IndexType;
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
      operation.firstReduction( sdata[ tid ], gid,                 input1, input2 );
      operation.firstReduction( sdata[ tid ], gid + gridSizeX,     input1, input2 );
      operation.firstReduction( sdata[ tid ], gid + 2 * gridSizeX, input1, input2 );
      operation.firstReduction( sdata[ tid ], gid + 3 * gridSizeX, input1, input2 );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size )
   {
      operation.firstReduction( sdata[ tid ], gid,                 input1, input2 );
      operation.firstReduction( sdata[ tid ], gid + gridSizeX,     input1, input2 );
      gid += 2 * gridSizeX;
   }
   while( gid < size )
   {
      operation.firstReduction( sdata[ tid ], gid,                 input1, input2 );
      gid += gridSizeX;
   }
   __syncthreads();


   //printf( "1: tid %d data %f \n", tid, sdata[ tid ] );

   /***
    *  Perform the parallel reduction.
    */
   if( blockSizeX >= 1024 ) {
      if( threadIdx.x < 512 ) {
         operation.commonReduction( sdata[ tid ], sdata[ tid + 512 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 512 ) {
      if( threadIdx.x < 256 ) {
         operation.commonReduction( sdata[ tid ], sdata[ tid + 256 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 256 ) {
      if( threadIdx.x < 128 ) {
         operation.commonReduction( sdata[ tid ], sdata[ tid + 128 ] );
      }
      __syncthreads();
   }
   if( blockSizeX >= 128 ) {
      if( threadIdx.x <  64 ) {
         operation.commonReduction( sdata[ tid ], sdata[ tid + 64 ] );
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
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 32 ] );
      }
      if( blockSizeX >= 32 ) {
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 16 ] );
      }
      if( blockSizeX >= 16 ) {
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 8 ] );
      }
      if( blockSizeX >=  8 ) {
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 4 ] );
      }
      if( blockSizeX >=  4 ) {
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 2 ] );
      }
      if( blockSizeX >=  2 ) {
         operation.commonReduction( vsdata[ tid ], vsdata[ tid + 1 ] );
      }
   }

   /***
    * Store the result back in the global memory.
    */
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x ] = sdata[ tid ];
   }
}

template< typename Operation, typename Index >
int
CudaMultireductionKernelLauncher( Operation& operation,
                                  const int n,
                                  const Index size,
                                  const typename Operation::DataType1* input1,
                                  const Index ldInput1,
                                  const typename Operation::DataType2* input2,
                                  typename Operation::ResultType*& output )
{
   typedef typename Operation::ResultType ResultType;

   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // Therefore,  desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
   // where blocksPerMultiprocessor is determined according to the number of
   // available registers on the multiprocessor.
   // On Tesla K40c, desGridSize = 8 * 15 = 120.
   const int activeDevice = Devices::CudaDeviceInfo::getActiveDevice();
   const int blocksdPerMultiprocessor = Devices::CudaDeviceInfo::getRegistersPerMultiprocessor( activeDevice )
                                      / ( Multireduction_maxThreadsPerBlock * Multireduction_registersPerThread );
   const int desGridSizeX = blocksdPerMultiprocessor * Devices::CudaDeviceInfo::getCudaMultiprocessors( activeDevice );
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
   cudaReductionBuffer.setSize( buf_size );
   output = cudaReductionBuffer.template getData< ResultType >();

   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   const Index shmem = (blockSize.x <= 32)
            ? 2 * blockSize.x * blockSize.y * sizeof( ResultType )
            : blockSize.x * blockSize.y * sizeof( ResultType );

   //cout << "Multireduction of " << n << " datasets, block size (" << blockSize.x << "," << blockSize.y << "), grid size (" << gridSize.x << "," << gridSize.y << "), shmem " << shmem <<std::endl;

   /***
    * Depending on the blockSize we generate appropriate template instance.
    */
   switch( blockSize.x )
   {
      case 512:
         CudaMultireductionKernel< 512 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case 256:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< 256, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< 256 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case 128:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< 128, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< 128 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  64:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  64, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  64 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  32:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  32, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  32 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case  16:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  16, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  16 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
     case   8:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   8, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   8 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case   4:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   4, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   4 >
        <<< gridSize, blockSize, shmem >>>( operation,  n,size, input1, ldInput1, input2, output);
        break;
      case   2:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   2, Operation, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   2 >
         <<< gridSize, blockSize, shmem >>>( operation, n, size, input1, ldInput1, input2, output);
         break;
      case   1:
         TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
      default:
         TNL_ASSERT( false, std::cerr << "Block size is " << blockSize.x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." << std::endl );
   }
   TNL_CHECK_CUDA_DEVICE;

   // return the size of the output array on the CUDA device
   return gridSize.x;
}
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
