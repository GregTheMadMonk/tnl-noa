#pragma once

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
CudaMultireductionKernel( Operation& operation,
                          const typename Operation::IndexType n,
                          const typename Operation::IndexType size,
                          const typename Operation::RealType* input1,
                          const typename Operation::IndexType ldInput1,
                          const typename Operation::RealType* input2,
                          typename Operation::ResultType* output )
{
   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::ResultType ResultType;

   extern __shared__ __align__ ( 8 ) char __sdata[];

   ResultType* sdata = reinterpret_cast< ResultType* >( __sdata );

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
#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
