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

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Containers/Algorithms/CudaReductionBuffer.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

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
static constexpr int Multireduction_registersPerThread = 32;   // empirically determined optimal value

// __CUDA_ARCH__ is defined only in device code!
#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 8;
#else
   static constexpr int Multireduction_minBlocksPerMultiprocessor = 4;
#endif

template< int blockSizeX,
          typename Result,
          typename DataFetcher,
          typename Reduction,
          typename Index >
__global__ void
__launch_bounds__( Multireduction_maxThreadsPerBlock, Multireduction_minBlocksPerMultiprocessor )
CudaMultireductionKernel( const Result zero,
                          DataFetcher dataFetcher,
                          const Reduction reduction,
                          const Index size,
                          const int n,
                          Result* output )
{
   Result* sdata = Devices::Cuda::getSharedMemory< Result >();

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.y * blockDim.x + threadIdx.x;
         Index gid = blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSizeX = blockDim.x * gridDim.x;

   // Get the dataset index.
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   if( y >= n ) return;

   sdata[ tid ] = zero;

   // Start with the sequential reduction and push the result into the shared memory.
   while( gid + 4 * gridSizeX < size ) {
      reduction( sdata[ tid ], dataFetcher( gid,                 y ) );
      reduction( sdata[ tid ], dataFetcher( gid + gridSizeX,     y ) );
      reduction( sdata[ tid ], dataFetcher( gid + 2 * gridSizeX, y ) );
      reduction( sdata[ tid ], dataFetcher( gid + 3 * gridSizeX, y ) );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size ) {
      reduction( sdata[ tid ], dataFetcher( gid, y ) );
      reduction( sdata[ tid ], dataFetcher( gid + gridSizeX, y ) );
      gid += 2 * gridSizeX;
   }
   while( gid < size ) {
      reduction( sdata[ tid ], dataFetcher( gid, y ) );
      gid += gridSizeX;
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSizeX >= 1024 ) {
      if( threadIdx.x < 512 )
         reduction( sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSizeX >= 512 ) {
      if( threadIdx.x < 256 )
         reduction( sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSizeX >= 256 ) {
      if( threadIdx.x < 128 )
         reduction( sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSizeX >= 128 ) {
      if( threadIdx.x <  64 )
         reduction( sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
   }

   // This runs in one warp so we use __syncwarp() instead of __syncthreads().
   if( threadIdx.x < 32 ) {
      if( blockSizeX >= 64 )
         reduction( sdata[ tid ], sdata[ tid + 32 ] );
      __syncwarp();
      // Note that here we do not have to check if tid < 16 etc, because we have
      // 2 * blockSize.x elements of shared memory per block, so we do not
      // access out of bounds. The results for the upper half will be undefined,
      // but unused anyway.
      if( blockSizeX >= 32 )
         reduction( sdata[ tid ], sdata[ tid + 16 ] );
      __syncwarp();
      if( blockSizeX >= 16 )
         reduction( sdata[ tid ], sdata[ tid + 8 ] );
      __syncwarp();
      if( blockSizeX >=  8 )
         reduction( sdata[ tid ], sdata[ tid + 4 ] );
      __syncwarp();
      if( blockSizeX >=  4 )
         reduction( sdata[ tid ], sdata[ tid + 2 ] );
      __syncwarp();
      if( blockSizeX >=  2 )
         reduction( sdata[ tid ], sdata[ tid + 1 ] );
   }

   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x + y * gridDim.x ] = sdata[ tid ];
   }
}
#endif

template< typename Result,
          typename DataFetcher,
          typename Reduction,
          typename Index >
int
CudaMultireductionKernelLauncher( const Result zero,
                                  DataFetcher dataFetcher,
                                  const Reduction reduction,
                                  const Index size,
                                  const int n,
                                  Result*& output )
{
#ifdef HAVE_CUDA
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
   blockSize.y = TNL::min( n, 16 );

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

   gridSize.x = TNL::min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSizeX );
   gridSize.y = Devices::Cuda::getNumberOfBlocks( n, blockSize.y );

   if( gridSize.y > (unsigned) Devices::Cuda::getMaxGridSize() ) {
      std::cerr << "Maximum gridSize.y limit exceeded (limit is 65535, attempted " << gridSize.y << ")." << std::endl;
      throw 1;
   }

   // create reference to the reduction buffer singleton and set size
   // (make an overestimate to avoid reallocation on every call if n is increased by 1 each time)
   const std::size_t buf_size = 8 * ( n / 8 + 1 ) * desGridSizeX * sizeof( Result );
   CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
   cudaReductionBuffer.setSize( buf_size );
   output = cudaReductionBuffer.template getData< Result >();

   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   const Index shmem = (blockSize.x <= 32)
            ? 2 * blockSize.x * blockSize.y * sizeof( Result )
            : blockSize.x * blockSize.y * sizeof( Result );

   // Depending on the blockSize we generate appropriate template instance.
   switch( blockSize.x )
   {
      case 512:
         CudaMultireductionKernel< 512 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case 256:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< 256, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< 256 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case 128:
         cudaFuncSetCacheConfig(CudaMultireductionKernel< 128, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel< 128 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case  64:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  64, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  64 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case  32:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  32, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  32 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case  16:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<  16, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<  16 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
     case   8:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   8, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   8 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case   4:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   4, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   4 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
        break;
      case   2:
         cudaFuncSetCacheConfig(CudaMultireductionKernel<   2, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

         CudaMultireductionKernel<   2 >
         <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, size, n, output );
         break;
      case   1:
         throw std::logic_error( "blockSize should not be 1." );
      default:
         throw std::logic_error( "Block size is " + std::to_string(blockSize.x) + " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
   }
   cudaStreamSynchronize(0);
   TNL_CHECK_CUDA_DEVICE;

   // return the size of the output array on the CUDA device
   return gridSize.x;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
