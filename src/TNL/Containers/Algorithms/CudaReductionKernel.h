/***************************************************************************
                          CudaReductionKernel.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>  // std::pair

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
static constexpr int Reduction_registersPerThread = 32;   // empirically determined optimal value

// __CUDA_ARCH__ is defined only in device code!
#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Reduction_minBlocksPerMultiprocessor = 8;
#else
   static constexpr int Reduction_minBlocksPerMultiprocessor = 4;
#endif

template< int blockSize,
   typename Result,
   typename DataFetcher,
   typename Reduction,
   typename VolatileReduction,
   typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionKernel( const Result zero,
                     const DataFetcher dataFetcher,
                     const Reduction reduction,
                     const VolatileReduction volatileReduction,
                     const Index size,
                     Result* output )
{
   using IndexType = Index;
   using ResultType = Result;

   ResultType* sdata = Devices::Cuda::getSharedMemory< ResultType >();

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const IndexType tid = threadIdx.x;
         IndexType gid = blockIdx.x * blockDim. x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   sdata[ tid ] = zero;

   // Read data into the shared memory. We start with the sequential reduction.
   while( gid + 4 * gridSize < size ) {
      reduction( sdata[ tid ], dataFetcher( gid ) );
      reduction( sdata[ tid ], dataFetcher( gid + gridSize ) );
      reduction( sdata[ tid ], dataFetcher( gid + 2 * gridSize ) );
      reduction( sdata[ tid ], dataFetcher( gid + 3 * gridSize ) );
      gid += 4 * gridSize;
   }
   while( gid + 2 * gridSize < size ) {
      reduction( sdata[ tid ], dataFetcher( gid ) );
      reduction( sdata[ tid ], dataFetcher( gid + gridSize ) );
      gid += 2 * gridSize;
   }
   while( gid < size ) {
      reduction( sdata[ tid ], dataFetcher( gid ) );
      gid += gridSize;
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSize >= 1024 ) {
      if( tid < 512 )
         reduction( sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSize >= 512 ) {
      if( tid < 256 )
         reduction( sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSize >= 256 ) {
      if( tid < 128 )
         reduction( sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSize >= 128 ) {
      if( tid <  64 )
         reduction( sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
   }

   // This runs in one warp so it is synchronized implicitly.
   if( tid < 32 ) {
      volatile ResultType* vsdata = sdata;
      if( blockSize >= 64 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 32 ] );
      }
      // Note that here we do not have to check if tid < 16 etc, because we have
      // twice as much shared memory, so we do not access out of bounds. The
      // results for the upper half will be undefined, but unused anyway.
      if( blockSize >= 32 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 16 ] );
      }
      if( blockSize >= 16 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 8 ] );
      }
      if( blockSize >=  8 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 4 ] );
      }
      if( blockSize >=  4 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 2 ] );
      }
      if( blockSize >=  2 ) {
         volatileReduction( vsdata[ tid ], vsdata[ tid + 1 ] );
      }
   }

   // Store the result back in the global memory.
   if( tid == 0 )
      output[ blockIdx.x ] = sdata[ 0 ];
}

template< int blockSize,
   typename Result,
   typename DataFetcher,
   typename Reduction,
   typename VolatileReduction,
   typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionWithArgumentKernel( const Result zero,
                                 const DataFetcher dataFetcher,
                                 const Reduction reduction,
                                 const VolatileReduction volatileReduction,
                                 const Index size,
                                 Result* output,
                                 Index* idxOutput,
                                 const Index* idxInput = nullptr )
{
   using IndexType = Index;
   using ResultType = Result;

   ResultType* sdata = Devices::Cuda::getSharedMemory< ResultType >();
   IndexType* sidx = reinterpret_cast< IndexType* >( &sdata[ blockDim.x ] );

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const IndexType tid = threadIdx.x;
         IndexType gid = blockIdx.x * blockDim. x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   // Read data into the shared memory. We start with the sequential reduction.
   if( idxInput ) {
      if( gid < size ) {
         sdata[ tid ] = dataFetcher( gid );
         sidx[ tid ] = idxInput[ gid ];
         gid += gridSize;
      } else {
         sdata[ tid ] = zero;
      }
      while( gid + 4 * gridSize < size ) {
         reduction( sidx[ tid ], idxInput[ gid ], sdata[ tid ], dataFetcher( gid ) );
         reduction( sidx[ tid ], idxInput[ gid + gridSize ], sdata[ tid ], dataFetcher( gid + gridSize ) );
         reduction( sidx[ tid ], idxInput[ gid + 2 * gridSize ], sdata[ tid ], dataFetcher( gid + 2 * gridSize ) );
         reduction( sidx[ tid ], idxInput[ gid + 3 * gridSize ], sdata[ tid ], dataFetcher( gid + 3 * gridSize ) );
         gid += 4 * gridSize;
      }
      while( gid + 2 * gridSize < size ) {
         reduction( sidx[ tid ], idxInput[ gid ], sdata[ tid ], dataFetcher( gid ) );
         reduction( sidx[ tid ], idxInput[ gid + gridSize ], sdata[ tid ], dataFetcher( gid + gridSize ) );
         gid += 2 * gridSize;
      }
      while( gid < size ) {
         reduction( sidx[ tid ], idxInput[ gid ], sdata[ tid ], dataFetcher( gid ) );
         gid += gridSize;
      }
   }
   else {
      if( gid < size ) {
         sdata[ tid ] = dataFetcher( gid );
         sidx[ tid ] = gid;
         gid += gridSize;
      } else {
         sdata[ tid ] = zero;
      }
      while( gid + 4 * gridSize < size ) {
         reduction( sidx[ tid ], gid, sdata[ tid ], dataFetcher( gid ) );
         reduction( sidx[ tid ], gid + gridSize, sdata[ tid ], dataFetcher( gid + gridSize ) );
         reduction( sidx[ tid ], gid + 2 * gridSize, sdata[ tid ], dataFetcher( gid + 2 * gridSize ) );
         reduction( sidx[ tid ], gid + 3 * gridSize, sdata[ tid ], dataFetcher( gid + 3 * gridSize ) );
         gid += 4 * gridSize;
      }
      while( gid + 2 * gridSize < size ) {
         reduction( sidx[ tid ], gid, sdata[ tid ], dataFetcher( gid ) );
         reduction( sidx[ tid ], gid + gridSize, sdata[ tid ], dataFetcher( gid + gridSize ) );
         gid += 2 * gridSize;
      }
      while( gid < size ) {
         reduction( sidx[ tid ], gid, sdata[ tid ], dataFetcher( gid ) );
         gid += gridSize;
      }
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSize >= 1024 ) {
      if( tid < 512 )
         reduction( sidx[ tid ], sidx[ tid + 512 ], sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSize >= 512 ) {
      if( tid < 256 )
         reduction( sidx[ tid ], sidx[ tid + 256 ], sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSize >= 256 ) {
      if( tid < 128 )
         reduction( sidx[ tid ], sidx[ tid + 128 ], sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSize >= 128 ) {
      if( tid <  64 )
         reduction( sidx[ tid ], sidx[ tid + 64 ], sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
   }

   // This runs in one warp so it is synchronized implicitly.
   if( tid < 32 ) {
      volatile ResultType* vsdata = sdata;
      volatile IndexType* vsidx = sidx;
      if( blockSize >= 64 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 32 ], vsdata[ tid ], vsdata[ tid + 32 ] );
      }
      // Note that here we do not have to check if tid < 16 etc, because we have
      // twice as much shared memory, so we do not access out of bounds. The
      // results for the upper half will be undefined, but unused anyway.
      if( blockSize >= 32 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 16 ], vsdata[ tid ], vsdata[ tid + 16 ] );
      }
      if( blockSize >= 16 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 8 ], vsdata[ tid ], vsdata[ tid + 8 ] );
      }
      if( blockSize >=  8 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 4 ], vsdata[ tid ], vsdata[ tid + 4 ] );
      }
      if( blockSize >=  4 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 2 ], vsdata[ tid ], vsdata[ tid + 2 ] );
      }
      if( blockSize >=  2 ) {
         volatileReduction( vsidx[ tid ], vsidx[ tid + 1 ], vsdata[ tid ], vsdata[ tid + 1 ] );
      }
   }

   // Store the result back in the global memory.
   if( tid == 0 ) {
      output[ blockIdx.x ] = sdata[ 0 ];
      idxOutput[ blockIdx.x ] = sidx[ 0 ];
   }
}


template< typename Index,
          typename Result >
struct CudaReductionKernelLauncher
{
   using IndexType = Index;
   using ResultType = Result;

   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // Therefore,  desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
   // where blocksPerMultiprocessor is determined according to the number of
   // available registers on the multiprocessor.
   // On Tesla K40c, desGridSize = 8 * 15 = 120.
   //
   // Update:
   // It seems to be better to map only one CUDA block per one multiprocessor or maybe
   // just slightly more. Therefore we omit blocksdPerMultiprocessor in the following.
   CudaReductionKernelLauncher( const Index size )
   : activeDevice( Devices::CudaDeviceInfo::getActiveDevice() ),
     blocksdPerMultiprocessor( Devices::CudaDeviceInfo::getRegistersPerMultiprocessor( activeDevice )
                               / ( Reduction_maxThreadsPerBlock * Reduction_registersPerThread ) ),
     //desGridSize( blocksdPerMultiprocessor * Devices::CudaDeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     desGridSize( Devices::CudaDeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     originalSize( size )
   {
   }

   template< typename DataFetcher,
             typename Reduction,
             typename VolatileReduction >
   int start( const Reduction& reduction,
              const VolatileReduction& volatileReduction,
              const DataFetcher& dataFetcher,
              const Result& zero,
              ResultType*& output )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * sizeof( ResultType );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< ResultType >();

      this->reducedSize = this->launch( originalSize, reduction, volatileReduction, dataFetcher, zero, output );
      return this->reducedSize;
   }

   template< typename DataFetcher,
             typename Reduction,
             typename VolatileReduction >
   int startWithArgument( const Reduction& reduction,
                          const VolatileReduction& volatileReduction,
                          const DataFetcher& dataFetcher,
                          const Result& zero,
                          ResultType*& output,
                          IndexType*& idxOutput )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * ( sizeof( ResultType ) + sizeof( IndexType ) );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< ResultType >();
      idxOutput = reinterpret_cast< IndexType* >( &output[ 2 * desGridSize ] );

      this->reducedSize = this->launchWithArgument( originalSize, reduction, volatileReduction, dataFetcher, zero, output, idxOutput, nullptr );
      return this->reducedSize;
   }

   template< typename Reduction,
             typename VolatileReduction >
   Result
   finish( const Reduction& reduction,
           const VolatileReduction& volatileReduction,
           const Result& zero )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      ResultType* input = cudaReductionBuffer.template getData< ResultType >();
      ResultType* output = &input[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( IndexType i ) { return input[ i ]; };
         this->reducedSize = this->launch( this->reducedSize, reduction, volatileReduction, copyFetch, zero, output );
         std::swap( input, output );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );

      // Copy result on CPU
      ResultType result;
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( &result, output, 1 );
      return result;
   }

   template< typename Reduction,
             typename VolatileReduction >
   std::pair< Index, Result >
   finishWithArgument( const Reduction& reduction,
                       const VolatileReduction& volatileReduction,
                       const Result& zero )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      ResultType* input = cudaReductionBuffer.template getData< ResultType >();
      ResultType* output = &input[ desGridSize ];
      IndexType* idxInput = reinterpret_cast< IndexType* >( &output[ desGridSize ] );
      IndexType* idxOutput = &idxInput[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( IndexType i ) { return input[ i ]; };
         this->reducedSize = this->launchWithArgument( this->reducedSize, reduction, volatileReduction, copyFetch, zero, output, idxOutput, idxInput );
         std::swap( input, output );
         std::swap( idxInput, idxOutput );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );
      std::swap( idxInput, idxOutput );

      ////
      // Copy result on CPU
      std::pair< Index, Result > result;
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( &result.first, idxOutput, 1 );
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( &result.second, output, 1 );
      return result;
   }


   protected:
      template< typename DataFetcher,
                typename Reduction,
                typename VolatileReduction >
      int launch( const Index size,
                  const Reduction& reduction,
                  const VolatileReduction& volatileReduction,
                  const DataFetcher& dataFetcher,
                  const Result& zero,
                  Result* output )
      {
         dim3 blockSize, gridSize;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = TNL::min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // when there is only one warp per blockSize.x, we need to allocate two warps
         // worth of shared memory so that we don't index shared memory out of bounds
         const IndexType shmem = (blockSize.x <= 32)
                  ? 2 * blockSize.x * sizeof( ResultType )
                  : blockSize.x * sizeof( ResultType );

        // This is "general", but this method always sets blockSize.x to a specific value,
        // so runtime switch is not necessary - it only prolongs the compilation time.
/*
        // Depending on the blockSize we generate appropriate template instance.
        switch( blockSize.x )
        {
           case 512:
              CudaReductionKernel< 512 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case 256:
              cudaFuncSetCacheConfig(CudaReductionKernel< 256, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel< 256 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case 128:
              cudaFuncSetCacheConfig(CudaReductionKernel< 128, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel< 128 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case  64:
              cudaFuncSetCacheConfig(CudaReductionKernel<  64, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<  64 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case  32:
              cudaFuncSetCacheConfig(CudaReductionKernel<  32, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<  32 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case  16:
              cudaFuncSetCacheConfig(CudaReductionKernel<  16, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<  16 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
          case   8:
              cudaFuncSetCacheConfig(CudaReductionKernel<   8, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<   8 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case   4:
              cudaFuncSetCacheConfig(CudaReductionKernel<   4, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<   4 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case   2:
              cudaFuncSetCacheConfig(CudaReductionKernel<   2, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionKernel<   2 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
              break;
           case   1:
              TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
           default:
              TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
        }
        TNL_CHECK_CUDA_DEVICE;
*/

        // Check just to future-proof the code setting blockSize.x
        if( blockSize.x == Reduction_maxThreadsPerBlock ) {
           cudaFuncSetCacheConfig(CudaReductionKernel< Reduction_maxThreadsPerBlock, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

           CudaReductionKernel< Reduction_maxThreadsPerBlock >
           <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output);
        }
        else {
           TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
        }

        // Return the size of the output array on the CUDA device
        return gridSize.x;
      }

      template< typename DataFetcher,
                typename Reduction,
                typename VolatileReduction >
      int launchWithArgument( const Index size,
                              const Reduction& reduction,
                              const VolatileReduction& volatileReduction,
                              const DataFetcher& dataFetcher,
                              const Result& zero,
                              Result* output,
                              Index* idxOutput,
                              const Index* idxInput )
      {
         dim3 blockSize, gridSize;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = TNL::min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // when there is only one warp per blockSize.x, we need to allocate two warps
         // worth of shared memory so that we don't index shared memory out of bounds
         const IndexType shmem = (blockSize.x <= 32)
                  ? 2 * blockSize.x * ( sizeof( ResultType ) + sizeof( Index ) )
                  : blockSize.x * ( sizeof( ResultType ) + sizeof( Index ) );

        // This is "general", but this method always sets blockSize.x to a specific value,
        // so runtime switch is not necessary - it only prolongs the compilation time.
/*
        // Depending on the blockSize we generate appropriate template instance.
        switch( blockSize.x )
        {
           case 512:
              CudaReductionWithArgumentKernel< 512 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case 256:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 256, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel< 256 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case 128:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 128, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel< 128 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case  64:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  64, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<  64 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case  32:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  32, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<  32 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case  16:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  16, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<  16 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
          case   8:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   8, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<   8 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case   4:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   4, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<   4 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case   2:
              cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   2, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

              CudaReductionWithArgumentKernel<   2 >
              <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
              break;
           case   1:
              TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
           default:
              TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
        }
        TNL_CHECK_CUDA_DEVICE;
*/

        // Check just to future-proof the code setting blockSize.x
        if( blockSize.x == Reduction_maxThreadsPerBlock ) {
           cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock, Result, DataFetcher, Reduction, VolatileReduction, Index >, cudaFuncCachePreferShared);

           CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock >
           <<< gridSize, blockSize, shmem >>>( zero, dataFetcher, reduction, volatileReduction, size, output, idxOutput, idxInput );
        }
        else {
           TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
        }

        // return the size of the output array on the CUDA device
        return gridSize.x;
      }


      const int activeDevice;
      const int blocksdPerMultiprocessor;
      const int desGridSize;
      const IndexType originalSize;
      IndexType reducedSize;
};
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
