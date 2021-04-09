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

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Algorithms/CudaReductionBuffer.h>
#include <TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Algorithms {
namespace detail {

/****
 * The performance of this kernel is very sensitive to register usage.
 * Compile with --ptxas-options=-v and configure these constants for given
 * architecture so that there are no local memory spills.
 */
static constexpr int Reduction_maxThreadsPerBlock = 256;  // must be a power of 2
static constexpr int Reduction_registersPerThread = 32;   // empirically determined optimal value

#ifdef HAVE_CUDA
// __CUDA_ARCH__ is defined only in device code!
#if (__CUDA_ARCH__ == 750 )
   // Turing has a limit of 1024 threads per multiprocessor
   static constexpr int Reduction_minBlocksPerMultiprocessor = 4;
#else
   static constexpr int Reduction_minBlocksPerMultiprocessor = 8;
#endif

/*
 * nvcc (as of 10.2) is totally fucked up, in some cases it does not recognize the
 * std::plus<void>::operator() function to be constexpr and hence __host__ __device__
 * (for example, when the arguments are StaticVector<3, double> etc). Hence, we use
 * this wrapper which triggers only a warning and not an error as is the case when
 * the reduction functor is called from a __global__ or __device__ function. Let's
 * hope it works otherwise...
 */
template< typename Reduction, typename Arg1, typename Arg2 >
__host__ __device__
auto CudaReductionFunctorWrapper( Reduction&& reduction, Arg1&& arg1, Arg2&& arg2 )
{
// let's suppress the aforementioned warning...
#ifdef __NVCC__
#pragma push
#pragma diag_suppress 2979  // error number for nvcc 10.2
#pragma diag_suppress 3123  // error number for nvcc 11.1
#endif
   return std::forward<Reduction>(reduction)( std::forward<Arg1>(arg1), std::forward<Arg2>(arg2) );
#ifdef __NVCC__
#pragma pop
#endif
}

template< int blockSize,
          typename Result,
          typename DataFetcher,
          typename Reduction,
          typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionKernel( const Result zero,
                     DataFetcher dataFetcher,
                     const Reduction reduction,
                     const Index begin,
                     const Index end,
                     Result* output )
{
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );
   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   constexpr int shmemElements = (blockSize <= 32) ? 2 * blockSize : blockSize;
   __shared__ Result sdata[shmemElements];

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.x;
         Index gid = begin + blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSize = blockDim.x * gridDim.x;

   sdata[ tid ] = zero;

   // Start with the sequential reduction and push the result into the shared memory.
   while( gid + 4 * gridSize < end ) {
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid ) );
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid + gridSize ) );
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid + 2 * gridSize ) );
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid + 3 * gridSize ) );
      gid += 4 * gridSize;
   }
   while( gid + 2 * gridSize < end ) {
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid ) );
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid + gridSize ) );
      gid += 2 * gridSize;
   }
   while( gid < end ) {
      sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], dataFetcher( gid ) );
      gid += gridSize;
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSize >= 1024 ) {
      if( tid < 512 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSize >= 512 ) {
      if( tid < 256 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSize >= 256 ) {
      if( tid < 128 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSize >= 128 ) {
      if( tid <  64 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
   }

   // This runs in one warp so we use __syncwarp() instead of __syncthreads().
   if( tid < 32 ) {
      if( blockSize >= 64 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 32 ] );
      __syncwarp();
      // Note that here we do not have to check if tid < 16 etc, because we have
      // 2 * blockSize.x elements of shared memory per block, so we do not
      // access out of bounds. The results for the upper half will be undefined,
      // but unused anyway.
      if( blockSize >= 32 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 16 ] );
      __syncwarp();
      if( blockSize >= 16 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 8 ] );
      __syncwarp();
      if( blockSize >=  8 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 4 ] );
      __syncwarp();
      if( blockSize >=  4 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 2 ] );
      __syncwarp();
      if( blockSize >=  2 )
         sdata[ tid ] = CudaReductionFunctorWrapper( reduction, sdata[ tid ], sdata[ tid + 1 ] );
   }

   // Store the result back in the global memory.
   if( tid == 0 )
      output[ blockIdx.x ] = sdata[ 0 ];
}

template< int blockSize,
          typename Result,
          typename DataFetcher,
          typename Reduction,
          typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionWithArgumentKernel( const Result zero,
                                 DataFetcher dataFetcher,
                                 const Reduction reduction,
                                 const Index begin,
                                 const Index end,
                                 Result* output,
                                 Index* idxOutput,
                                 const Index* idxInput = nullptr )
{
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );
   // when there is only one warp per blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   constexpr int shmemElements = (blockSize <= 32) ? 2 * blockSize : blockSize;
   __shared__ Result sdata[shmemElements];
   __shared__ Index sidx[shmemElements];

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.x;
         Index gid = begin + blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSize = blockDim.x * gridDim.x;

   // Start with the sequential reduction and push the result into the shared memory.
   if( idxInput ) {
      if( gid < end ) {
         sdata[ tid ] = dataFetcher( gid );
         sidx[ tid ] = idxInput[ gid ];
         gid += gridSize;
      } else {
         sdata[ tid ] = zero;
      }
      while( gid + 4 * gridSize < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], idxInput[ gid ] );
         reduction( sdata[ tid ], dataFetcher( gid + gridSize ), sidx[ tid ], idxInput[ gid + gridSize ] );
         reduction( sdata[ tid ], dataFetcher( gid + 2 * gridSize ), sidx[ tid ], idxInput[ gid + 2 * gridSize ] );
         reduction( sdata[ tid ], dataFetcher( gid + 3 * gridSize ), sidx[ tid ], idxInput[ gid + 3 * gridSize ] );
         gid += 4 * gridSize;
      }
      while( gid + 2 * gridSize < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], idxInput[ gid ] );
         reduction( sdata[ tid ], dataFetcher( gid + gridSize ), sidx[ tid ], idxInput[ gid + gridSize ] );
         gid += 2 * gridSize;
      }
      while( gid < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], idxInput[ gid ] );
         gid += gridSize;
      }
   }
   else {
      if( gid < end ) {
         sdata[ tid ] = dataFetcher( gid );
         sidx[ tid ] = gid;
         gid += gridSize;
      } else {
         sdata[ tid ] = zero;
      }
      while( gid + 4 * gridSize < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], gid );
         reduction( sdata[ tid ], dataFetcher( gid + gridSize ), sidx[ tid ], gid + gridSize );
         reduction( sdata[ tid ], dataFetcher( gid + 2 * gridSize ), sidx[ tid ], gid + 2 * gridSize );
         reduction( sdata[ tid ], dataFetcher( gid + 3 * gridSize ), sidx[ tid ], gid + 3 * gridSize );
         gid += 4 * gridSize;
      }
      while( gid + 2 * gridSize < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], gid );
         reduction( sdata[ tid ], dataFetcher( gid + gridSize ), sidx[ tid ], gid + gridSize );
         gid += 2 * gridSize;
      }
      while( gid < end ) {
         reduction( sdata[ tid ], dataFetcher( gid ), sidx[ tid ], gid );
         gid += gridSize;
      }
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSize >= 1024 ) {
      if( tid < 512 )
         reduction( sdata[ tid ], sdata[ tid + 512 ], sidx[ tid ], sidx[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSize >= 512 ) {
      if( tid < 256 )
         reduction( sdata[ tid ], sdata[ tid + 256 ], sidx[ tid ], sidx[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSize >= 256 ) {
      if( tid < 128 )
         reduction( sdata[ tid ], sdata[ tid + 128 ], sidx[ tid ], sidx[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSize >= 128 ) {
      if( tid <  64 )
         reduction( sdata[ tid ], sdata[ tid + 64 ], sidx[ tid ], sidx[ tid + 64 ] );
      __syncthreads();
   }

   // This runs in one warp so we use __syncwarp() instead of __syncthreads().
   if( tid < 32 ) {
      if( blockSize >= 64 )
         reduction( sdata[ tid ], sdata[ tid + 32 ], sidx[ tid ], sidx[ tid + 32 ] );
      __syncwarp();
      // Note that here we do not have to check if tid < 16 etc, because we have
      // 2 * blockSize.x elements of shared memory per block, so we do not
      // access out of bounds. The results for the upper half will be undefined,
      // but unused anyway.
      if( blockSize >= 32 )
         reduction( sdata[ tid ], sdata[ tid + 16 ], sidx[ tid ], sidx[ tid + 16 ] );
      __syncwarp();
      if( blockSize >= 16 )
         reduction( sdata[ tid ], sdata[ tid + 8 ], sidx[ tid ], sidx[ tid + 8 ] );
      __syncwarp();
      if( blockSize >=  8 )
         reduction( sdata[ tid ], sdata[ tid + 4 ], sidx[ tid ], sidx[ tid + 4 ] );
      __syncwarp();
      if( blockSize >=  4 )
         reduction( sdata[ tid ], sdata[ tid + 2 ], sidx[ tid ], sidx[ tid + 2 ] );
      __syncwarp();
      if( blockSize >=  2 )
         reduction( sdata[ tid ], sdata[ tid + 1 ], sidx[ tid ], sidx[ tid + 1 ] );
   }

   // Store the result back in the global memory.
   if( tid == 0 ) {
      output[ blockIdx.x ] = sdata[ 0 ];
      idxOutput[ blockIdx.x ] = sidx[ 0 ];
   }
}
#endif


template< typename Index,
          typename Result >
struct CudaReductionKernelLauncher
{
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
   CudaReductionKernelLauncher( const Index begin, const Index end )
   : activeDevice( Cuda::DeviceInfo::getActiveDevice() ),
     blocksdPerMultiprocessor( Cuda::DeviceInfo::getRegistersPerMultiprocessor( activeDevice )
                               / ( Reduction_maxThreadsPerBlock * Reduction_registersPerThread ) ),
     //desGridSize( blocksdPerMultiprocessor * Cuda::DeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     desGridSize( Cuda::DeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     begin( begin ), end( end )
   {
   }

   template< typename DataFetcher,
             typename Reduction >
   int start( const Reduction& reduction,
              DataFetcher& dataFetcher,
              const Result& zero,
              Result*& output )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * sizeof( Result );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< Result >();

      this->reducedSize = this->launch( begin, end, reduction, dataFetcher, zero, output );
      return this->reducedSize;
   }

   template< typename DataFetcher,
             typename Reduction >
   int startWithArgument( const Reduction& reduction,
                          DataFetcher& dataFetcher,
                          const Result& zero,
                          Result*& output,
                          Index*& idxOutput )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * ( sizeof( Result ) + sizeof( Index ) );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< Result >();
      idxOutput = reinterpret_cast< Index* >( &output[ 2 * desGridSize ] );

      this->reducedSize = this->launchWithArgument( begin, end, reduction, dataFetcher, zero, output, idxOutput, nullptr );
      return this->reducedSize;
   }

   template< typename Reduction >
   Result
   finish( const Reduction& reduction,
           const Result& zero )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( Index i ) { return input[ i ]; };
         this->reducedSize = this->launch( 0, this->reducedSize, reduction, copyFetch, zero, output );
         std::swap( input, output );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );

      // Copy result on CPU
      Result result;
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result, output, 1 );
      return result;
   }

   template< typename Reduction >
   std::pair< Result, Index >
   finishWithArgument( const Reduction& reduction,
                       const Result& zero )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];
      Index* idxInput = reinterpret_cast< Index* >( &output[ desGridSize ] );
      Index* idxOutput = &idxInput[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( Index i ) { return input[ i ]; };
         this->reducedSize = this->launchWithArgument( ( Index ) 0, this->reducedSize, reduction, copyFetch, zero, output, idxOutput, idxInput );
         std::swap( input, output );
         std::swap( idxInput, idxOutput );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );
      std::swap( idxInput, idxOutput );

      ////
      // Copy result on CPU
      std::pair< Result, Index > result;
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result.first, output, 1 );
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result.second, idxOutput, 1 );
      return result;
   }


   protected:
      template< typename DataFetcher,
                typename Reduction >
      int launch( const Index begin,
                  const Index end,
                  const Reduction& reduction,
                  DataFetcher& dataFetcher,
                  const Result& zero,
                  Result* output )
      {
#ifdef HAVE_CUDA
         const Index size = end - begin;
         dim3 blockSize, gridSize;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = TNL::min( Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // This is "general", but this method always sets blockSize.x to a specific value,
         // so runtime switch is not necessary - it only prolongs the compilation time.
/*
         // Depending on the blockSize we generate appropriate template instance.
         switch( blockSize.x )
         {
            case 512:
               CudaReductionKernel< 512 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case 256:
               cudaFuncSetCacheConfig(CudaReductionKernel< 256, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel< 256 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case 128:
               cudaFuncSetCacheConfig(CudaReductionKernel< 128, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel< 128 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case  64:
               cudaFuncSetCacheConfig(CudaReductionKernel<  64, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  64 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case  32:
               cudaFuncSetCacheConfig(CudaReductionKernel<  32, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  32 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case  16:
               cudaFuncSetCacheConfig(CudaReductionKernel<  16, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  16 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
           case   8:
               cudaFuncSetCacheConfig(CudaReductionKernel<   8, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   8 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case   4:
               cudaFuncSetCacheConfig(CudaReductionKernel<   4, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   4 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case   2:
               cudaFuncSetCacheConfig(CudaReductionKernel<   2, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   2 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output);
               break;
            case   1:
               TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
            default:
               TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
         }
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;
*/

         // Check just to future-proof the code setting blockSize.x
         if( blockSize.x == Reduction_maxThreadsPerBlock ) {
            cudaFuncSetCacheConfig(CudaReductionKernel< Reduction_maxThreadsPerBlock, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

            // shared memory is allocated statically inside the kernel
            CudaReductionKernel< Reduction_maxThreadsPerBlock >
            <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, begin, end, output);
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
         else {
            TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
         }

         // Return the size of the output array on the CUDA device
         return gridSize.x;
#else
         throw Exceptions::CudaSupportMissing();
#endif
      }

      template< typename DataFetcher,
                typename Reduction >
      int launchWithArgument( const Index begin,
                              const Index end,
                              const Reduction& reduction,
                              DataFetcher& dataFetcher,
                              const Result& zero,
                              Result* output,
                              Index* idxOutput,
                              const Index* idxInput )
      {
#ifdef HAVE_CUDA
         dim3 blockSize, gridSize;
         const Index size = end - begin;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = TNL::min( Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // This is "general", but this method always sets blockSize.x to a specific value,
         // so runtime switch is not necessary - it only prolongs the compilation time.
/*
         // Depending on the blockSize we generate appropriate template instance.
         switch( blockSize.x )
         {
            case 512:
               CudaReductionWithArgumentKernel< 512 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case 256:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 256, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel< 256 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case 128:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 128, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel< 128 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case  64:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  64, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  64 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case  32:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  32, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  32 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case  16:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  16, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  16 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
           case   8:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   8, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   8 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case   4:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   4, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   4 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case   2:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   2, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   2 >
               <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, size, output, idxOutput, idxInput );
               break;
            case   1:
               TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
            default:
               TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
         }
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;
*/

         // Check just to future-proof the code setting blockSize.x
         if( blockSize.x == Reduction_maxThreadsPerBlock ) {
            cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock, Result, DataFetcher, Reduction, Index >, cudaFuncCachePreferShared);

            // shared memory is allocated statically inside the kernel
            CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock >
            <<< gridSize, blockSize >>>( zero, dataFetcher, reduction, begin, end, output, idxOutput, idxInput );
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
         else {
            TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
         }

         // return the size of the output array on the CUDA device
         return gridSize.x;
#else
         throw Exceptions::CudaSupportMissing();
#endif
      }


      const int activeDevice;
      const int blocksdPerMultiprocessor;
      const int desGridSize;
      //const Index originalSize;
      const Index begin, end;
      Index reducedSize;
};

} // namespace detail
} // namespace Algorithms
} // namespace TNL
