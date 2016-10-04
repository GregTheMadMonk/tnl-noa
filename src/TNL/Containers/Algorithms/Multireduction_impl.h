#pragma once

//#define CUDA_REDUCTION_PROFILING

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/reduction-operations.h>
#include <TNL/Containers/ArrayOperations.h>
#include <TNL/Math.h>
#include <TNL/Containers/Algorithms/CudaReductionBuffer.h>
#include <TNL/Containers/Algorithms/CudaMultireductionKernel.h>
#include <TNL/Devices/CudaDeviceInfo.h>

#ifdef CUDA_REDUCTION_PROFILING
#include <TNL/Timer.h>
#include <iostream>
#endif

namespace TNL {
namespace Containers {
namespace Algorithms {

/****
 * Arrays smaller than the following constant are reduced on CPU.
 */
//static constexpr int Multireduction_minGpuDataSize = 16384;//65536; //16384;//1024;//256;
// TODO: benchmarks with different values
static constexpr int Multireduction_minGpuDataSize = 256;//65536; //16384;//1024;//256;

#ifdef HAVE_CUDA
template< typename Operation >
typename Operation::IndexType
multireduceOnCudaDevice( Operation& operation,
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

   if( gridSize.y > Devices::Cuda::getMaxGridSize() ) {
      std::cerr << "Maximum gridSize.y limit exceeded (limit is 65535, attempted " << gridSize.y << ")." << std::endl;
      throw 1;
   }

   // create reference to the reduction buffer singleton and set default size
   // (make an overestimate to avoid reallocation on every call if n is increased by 1 each time)
   const size_t buf_size = 8 * ( n / 8 + 1 ) * desGridSizeX * sizeof( ResultType );
   CudaReductionBuffer & cudaReductionBuffer = CudaReductionBuffer::getInstance();
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
         Assert( false, std::cerr << "blockSize should not be 1." << std::endl );
      default:
         Assert( false, std::cerr << "Block size is " << blockSize.x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." << std::endl );
   }
   checkCudaDevice;
   return gridSize.x;
}
#endif

/*
 * Parameters:
 *    operation: the operation used for reduction
 *    n: number of datasets to be reduced
 *    size: the size of each dataset
 *    deviceInput1: input array of size = n * ldInput1
 *    ldInput1: leading dimension of the deviceInput1 array
 *    deviceInput2: either nullptr or input array of size = size
 *    hostResult: output array of size = n
 */
template< typename Operation >
bool multireductionOnCudaDevice( Operation& operation,
                                 int n,
                                 const typename Operation::IndexType size,
                                 const typename Operation::RealType* deviceInput1,
                                 const typename Operation::IndexType ldInput1,
                                 const typename Operation::RealType* deviceInput2,
                                 typename Operation::ResultType* hostResult )
{
#ifdef HAVE_CUDA
   Assert( n > 0, );

   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;
   typedef typename Operation::ResultType ResultType;
   typedef typename Operation::LaterReductionOperation LaterReductionOperation;

   /***
    * First check if the input array(s) is/are large enough for the multireduction on GPU.
    * Otherwise copy it/them to host and multireduce on CPU.
    */
   if( n * ldInput1 < Multireduction_minGpuDataSize ) {
      RealType hostArray1[ Multireduction_minGpuDataSize ];
      // FIXME: hostArray2 is left undefined if deviceInput2 is nullptr
      RealType hostArray2[ Multireduction_minGpuDataSize ];
      if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( hostArray1, deviceInput1, n * ldInput1 ) )
         return false;
      if( deviceInput2 &&
          ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( hostArray2, deviceInput2, n * size ) )
         return false;
      return multireductionOnHostDevice( operation, n, size, hostArray1, ldInput1, hostArray2, hostResult );
   }

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   /****
    * Reduce the data on the CUDA device.
    */
   ResultType* deviceAux1 = nullptr;
   const IndexType reducedSize = multireduceOnCudaDevice( operation,
                                                          n,
                                                          size,
                                                          deviceInput1,
                                                          ldInput1,
                                                          deviceInput2,
                                                          deviceAux1 );
   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      cout << "   Multireduction of " << n << " datasets on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << endl;
      timer.reset();
      timer.start();
   #endif

   /***
    * Transfer the reduced data from device to host.
    */
   ResultType resultArray[ n * reducedSize ];
   if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< ResultType, ResultType, IndexType >( resultArray, deviceAux1, n * reducedSize ) )
      return false;

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << endl;
      timer.reset();
      timer.start();
   #endif

//   cout << "resultArray = [";
//   for( int i = 0; i < n * reducedSize; i++ ) {
//      cout << resultArray[ i ];
//      if( i < n * reducedSize - 1 )
//         cout << ", ";
//   }
//   cout << "]" << endl;

   /***
    * Reduce the data on the host system.
    */
   LaterReductionOperation laterReductionOperation;
   multireductionOnHostDevice( laterReductionOperation, n, reducedSize, resultArray, reducedSize, (RealType*) nullptr, hostResult );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      cout << "   Multireduction of small data set on CPU took " << timer.getRealTime() << " sec. " << endl;
   #endif

   return checkCudaDevice;
#else
   CudaSupportMissingMessage;
   return false;
#endif
};

/*
 * Parameters:
 *    operation: the operation used for reduction
 *    n: number of datasets to be reduced
 *    size: the size of each dataset
 *    input1: input array of size = n * ldInput1
 *    ldInput1: leading dimension of the input1 array
 *    input2: either nullptr or input array of size = size
 *    hostResult: output array of size = n
 */
template< typename Operation >
bool multireductionOnHostDevice( Operation& operation,
                                 int n,
                                 const typename Operation::IndexType size,
                                 const typename Operation::RealType* input1,
                                 const typename Operation::IndexType ldInput1,
                                 const typename Operation::RealType* input2,
                                 typename Operation::ResultType* result )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;

   for( int k = 0; k < n; k++ ) {
      result[ k ] = operation.initialValue();
      const RealType* _input1 = input1 + k * ldInput1;
      for( IndexType i = 0; i < size; i++ )
         result[ k ] = operation.reduceOnHost( i, result[ k ], _input1, input2 );
   }

   return true;
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
