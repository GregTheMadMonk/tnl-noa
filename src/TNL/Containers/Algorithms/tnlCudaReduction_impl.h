/***************************************************************************
                          tnlCudaReduction_impl.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Operation, int blockSize >
__device__
void
tnlCudaReduction< Operation, blockSize >::
reduce( Operation& operation,
        const IndexType size,
        const RealType* input1,
        const RealType* input2,
        ResultType* output )
{
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

#ifdef UNDEF

template< typename Real, typename Index, int blockSize >
__device__
void
tnlCudaReduction< tnlParallelReductionScalarProduct< Real, Index >, blockSize >::
reduce( Operation& operation,
        const IndexType size,
        const RealType* input1,
        const RealType* input2,
        ResultType* output )
{
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

   /***
    * Read data into the shared memory. We start with the
    * sequential reduction.
    */
   sdata[ tid ] = ( RealType ) 0;
   /*while( gid + 4 * gridSize < size )
   {
      sdata[ tid ] += input1[ gid                ] * input2[ gid ];
      sdata[ tid ] += input1[ gid + gridSize     ] * input2[ gid + gridSize ];
      sdata[ tid ] += input1[ gid + 2 * gridSize ] * input2[ gid + 2 * gridSize ];
      sdata[ tid ] += input1[ gid + 3 * gridSize ] * input2[ gid + 3 * gridSize ];
      gid += 4*gridSize;
   }
   while( gid + 2 * gridSize < size )
   {
      sdata[ tid ] += input1[ gid            ] * input2[ gid ];
      sdata[ tid ] += input1[ gid + gridSize ] * input2[ gid + gridSize ];
      gid += 2*gridSize;
   }*/
   while( gid < size )
   {
      sdata[ tid ] += input1[ gid ] * input2[ gid ];
      gid += gridSize;
   }
   __syncthreads();

   //printf( "1: tid %d data %f \n", tid, sdata[ tid ] );

   /***
    *  Perform the parallel reduction.
    */
   if( blockSize >= 1024 )
   {
      if( tid < 512 )
         sdata[ tid ] += sdata[ tid + 512 ];
      __syncthreads();
   }
   if( blockSize >= 512 )
   {
      if( tid < 256 )
         sdata[ tid ] += sdata[ tid + 256 ];
      __syncthreads();
   }
   if( blockSize >= 256 )
   {
      if( tid < 128 )
         sdata[ tid ] += sdata[ tid + 128 ];
      __syncthreads();
      //printf( "2: tid %d data %f \n", tid, sdata[ tid ] );
   }

   if( blockSize >= 128 )
   {
      if( tid <  64 )
         sdata[ tid ] += sdata[ tid + 64 ];
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
         vsdata[ tid ] += vsdata[ tid + 32 ];
         //__syncthreads();
         //printf( "4: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >= 32 )
      {
         vsdata[ tid ] += vsdata[ tid + 16 ];
         //__syncthreads();
         //printf( "5: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >= 16 )
      {
         vsdata[ tid ] += vsdata[ tid + 8 ];
         //__syncthreads();
         //printf( "6: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  8 )
      {
         vsdata[ tid ] += vsdata[ tid + 4 ];
         //__syncthreads();
         //printf( "7: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  4 )
      {
         vsdata[ tid ] += vsdata[ tid + 2 ];
         //__syncthreads();
         //printf( "8: tid %d data %f \n", tid, sdata[ tid ] );
      }
      if( blockSize >=  2 )
      {
         vsdata[ tid ] += vsdata[ tid + 1 ];
         //__syncthreads();
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

#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

