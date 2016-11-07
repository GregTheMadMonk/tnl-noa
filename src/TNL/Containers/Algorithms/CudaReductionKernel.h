/***************************************************************************
                          CudaReductionKernel.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {
namespace Algorithms {

#ifdef HAVE_CUDA

template< typename Operation, int blockSize >
__global__
void
CudaReductionKernel( Operation& operation,
                     const typename Operation::IndexType size,
                     const typename Operation::RealType* input1,
                     const typename Operation::RealType* input2,
                     typename Operation::ResultType* output )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::ResultType ResultType;

   extern __shared__ __align__ ( 8 ) char __sdata[];

   ResultType* sdata = reinterpret_cast< ResultType* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * gridSize is the number of element processed by all blocks at the
    * same time.
    */
   IndexType tid = threadIdx. x;
   IndexType gid = blockIdx. x * blockDim. x + threadIdx. x;
   IndexType gridSize = blockDim. x * gridDim.x;

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
      gid += 4*gridSize;
   }
   while( gid + 2 * gridSize < size )
   {
      operation.cudaFirstReduction( sdata[ tid ], gid,                input1, input2 );
      operation.cudaFirstReduction( sdata[ tid ], gid + gridSize,     input1, input2 );
      gid += 2*gridSize;
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

   // TODO: optimize similarly to multireduction
   const int minGPUReductionDataSize = 256;
   const IndexType desGridSize( minGPUReductionDataSize );
   dim3 blockSize( 256 ), gridSize( 0 );
   gridSize.x = min( Devices::Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

   // create reference to the reduction buffer singleton and set default size
   CudaReductionBuffer & cudaReductionBuffer = CudaReductionBuffer::getInstance( 8 * minGPUReductionDataSize );

   if( ! cudaReductionBuffer.setSize( gridSize.x * sizeof( ResultType ) ) )
      return false;
   output = cudaReductionBuffer.template getData< ResultType >();
   IndexType shmem = blockSize.x * sizeof( ResultType );

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
         CudaReductionKernel< Operation, 256 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case 128:
         CudaReductionKernel< Operation, 128 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  64:
         CudaReductionKernel< Operation,  64 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  32:
         CudaReductionKernel< Operation,  32 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case  16:
         CudaReductionKernel< Operation,  16 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
     case   8:
         CudaReductionKernel< Operation,   8 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case   4:
         CudaReductionKernel< Operation,   4 >
        <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
        break;
      case   2:
         CudaReductionKernel< Operation,   2 >
         <<< gridSize, blockSize, shmem >>>( operation, size, input1, input2, output);
         break;
      case   1:
         Assert( false, std::cerr << "blockSize should not be 1." << std::endl );
      default:
         Assert( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
   }
   //checkCudaDevice;
   return gridSize. x;
}
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
