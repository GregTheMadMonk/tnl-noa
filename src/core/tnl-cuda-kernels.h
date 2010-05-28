/***************************************************************************
                          tnl-cuda-kernels.h
                             -------------------
    begin                : Jan 14, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDAKERNELS_H_
#define TNLCUDAKERNELS_H_

#include <core/tnlAssert.h>
#include <core/tnlLongVectorCUDA.h>

using namespace std;

enum tnlOperation { tnlMin, tnlMax, tnlSum };

#ifdef HAVE_CUDA

/****
 * The following kernels and functions have been adopted from the diploma work of Jan Vacata.
 * Vacata Jan, GPGPU: General Purpose Computation on GPUs, diploma thesis,
 *  Department of mathematics, FNSPE, CTU in Prague, 2008.
 *
 * His code was extended by Tomas Oberhuber even for data arrays with size different from
 * a power of 2.
 *
 * For the educative and also testing/debuging reasons we have 6 version of this algorithm here.
 * Version 1 is the slowest and version 6 is the fastest - teste on CUDA architecture 1.0 - 1.3.
 * Another improvements are possible for the future devices.
 *
 */

/***
 * This function returns minimum of two numbers stored on the device.
 */
template< class T > __device__ T tnlCudaMin( const T& a,
                                             const T& b )
{
   return a < b ? a : b;
}

__device__ int tnlCudaMin( const int& a,
                           const int& b )
{
   return min( a, b );
}

__device__ float tnlCudaMin( const float& a,
                             const float& b )
{
   return fminf( a, b );
}

__device__ double tnlCudaMin( const double& a,
                              const double& b )
{
   return fmin( a, b );
}

/***
 * This function returns maximum of two numbers stored on the device.
 */
template< class T > __device__ T tnlCudaMax( const T& a,
                                             const T& b )
{
   return a > b ? a : b;
}

__device__ int tnlCudaMax( const int& a,
                           const int& b )
{
   return max( a, b );
}

__device__ float tnlCudaMax( const float& a,
                             const float& b )
{
   return fmaxf( a, b );
}

__device__ double tnlCudaMax( const double& a,
                              const double& b )
{
   return fmax( a, b );
}


/***
 * For each thread in block with thread ID smaller then s this function reduces
 * data elements with indecis tid and tid + s. Here we assume that for each
 * tid the tid + s element also exists i.e. we have even number of elements.
 */
template< class T, tnlOperation operation >
__device__ void reduceAligned( unsigned int tid,
                               unsigned int s,
                               T* sdata )
{
   if( tid < s )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlSum )
         sdata[ tid ] += sdata[ tid + s ];
   }
}

/***
 * For each thread in block with thread ID smaller then s this function reduces
 * data elements with indecis tid and tid + s. This is a modified version of
 * the previous algorithm. Thid one works even for odd number of elements but
 * it is a bit slower.
 */
template< class T, tnlOperation operation >
__device__ void reduceNonAligned( unsigned int tid,
                                  unsigned int s,
                                  unsigned int n,
                                  T* sdata )
{
   if( tid < s )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlSum )
         sdata[ tid ] += sdata[ tid + s ];
   }
   /* This is for the case when we have odd number of elements.
    * The last one will be reduced using the thread with ID 0.
    */
   if( s > 32 )
      __syncthreads();
   if( 2 * s < n && tid == n - 1 )
   {
      if( operation == tnlMin )
         sdata[ 0 ] = tnlCudaMin( sdata[ 0 ], sdata[ tid ] );
      if( operation == tnlMax )
         sdata[ 0 ] = tnlCudaMax( sdata[ 0 ], sdata[ tid ] );
      if( operation == tnlSum )
         sdata[ 0 ] += sdata[ tid ];
   }
}


/***
 * The final parallel reduction - version 6.
 * In comparison to the version 5 we have unrolled the reduction loop.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction2
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation, int blockSize >
__global__ void tnlCUDAReductionKernel( const int size,
                                        const T* deviceInput,
                                        T* deviceOutput,
                                        T* dbg_array1 = 0 )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id in this block.
    * gridSize is the number of element processed by all blocks at the
    * same time.
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
   unsigned int lastTId = size - 2 * blockIdx. x * blockDim. x;
   unsigned int gridSize = 2 * blockDim. x * gridDim.x;

   /***
    * Read data into the shared memory. We start with the
    * sequential reduction. This modification gives the
    * greatest performance improvement :-).
    */
   if( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlSum )
         sdata[ tid ] = deviceInput[ gid ] + deviceInput[ gid + blockDim. x ];
   }
   else if( gid < size )
   {
      sdata[ tid ] = deviceInput[ gid ];
   }
   gid += gridSize;
   while( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], :: tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], :: tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlSum )
         sdata[ tid ] += deviceInput[gid] + deviceInput[ gid + blockDim. x ];
      gid += gridSize;
   }
   if( gid < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlSum )
         sdata[ tid ] += deviceInput[gid];
   }
   __syncthreads();


   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which is one half of the elements to reduce.
    *  Each thread with ID < s reduce elements tid and tid + s. The result is stored
    *  in shared memory in sdata 0 .. s. We set s = s / 2 ( i.e. s >>= 1) and repeat
    *  the algorithm again until s = 1.
    *  We also separate the case when the blockDim. x is power of 2 and the algorithm
    *  can be written in more efficient way without some conditions.
    *  The loop which was in the version 5 is unrolled here.
    */
   unsigned int n = lastTId < blockDim. x ? lastTId : blockDim. x;
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
   {
      if( blockSize >= 512 )
      {
         if( tid < 256 )
            reduceAligned< T, operation >( tid, 256, sdata );
         __syncthreads();
      }
      if( blockSize >= 256 )
      {
         if( tid < 128 )
            reduceAligned< T, operation >( tid, 128, sdata );
         __syncthreads();
      }
      if( blockSize >= 128 )
      {
         if( tid <  64 )
            reduceAligned< T, operation >( tid, 64, sdata );
         __syncthreads();
      }

      /***
       * This runs in one warp so it is synchronised implicitly.
       */
      if (tid < 32)
      {
         if( blockSize >= 64 )
            reduceAligned< T, operation >( tid, 32, sdata );
         if( blockSize >= 32 )
            reduceAligned< T, operation >( tid, 16, sdata );
         if( blockSize >= 16 )
            reduceAligned< T, operation >( tid,  8, sdata );
         if( blockSize >=  8 )
            reduceAligned< T, operation >( tid,  4, sdata );
         if( blockSize >=  4 )
            reduceAligned< T, operation >( tid,  2, sdata );
         if( blockSize >=  2 )
            reduceAligned< T, operation >( tid,  1, sdata );
      }
   }
   else
   {
      unsigned int s;
      if( n >= 512 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 256 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 128 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 64 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 32 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      /***
       * This runs in one warp so it is synchronised implicitly.
       */
      if( n >= 16 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
      }
      if( n >= 8 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
      }
      if( n >= 4 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
      }
      if( n >= 2 )
      {
         s = n / 2;
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
      }

   }

   /***
    * Store the result back in the global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling the final CUDA kernel (version 6) for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 */
template< class T, tnlOperation operation >
bool tnlCUDAReduction( const int size,
                       const T* deviceInput,
                       T& result,
                       T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    * @param desGridSize is desired grid size
    */
   const int desBlockSize = 128;
   const int desGridSize = 2048;

   T* dbg_array1; // debuging array

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      gridSize. x = :: Min( ( int ) ( sizeReduced / blockSize. x + 1 ) / 2, desGridSize );

      /***
       * We align the blockSize to the power of 2.
       */
      int alignedBlockSize = 1;
      while( alignedBlockSize < blockSize. x ) alignedBlockSize <<= 1;
      blockSize. x = alignedBlockSize;
      int shmem = blockSize. x * sizeof( T );
      /***
       * Depending on the blockSize we generate appropriate template instance.
       */
      switch( blockSize. x )
      {
         case 512:
            tnlCUDAReductionKernel< T, operation, 512 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case 256:
            tnlCUDAReductionKernel< T, operation, 256 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case 128:
            tnlCUDAReductionKernel< T, operation, 128 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case  64:
            tnlCUDAReductionKernel< T, operation,  64 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case  32:
            tnlCUDAReductionKernel< T, operation,  32 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case  16:
            tnlCUDAReductionKernel< T, operation,  16 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case   8:
            tnlCUDAReductionKernel< T, operation,   8 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case   4:
            tnlCUDAReductionKernel< T, operation,   4 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case   2:
            tnlCUDAReductionKernel< T, operation,   2 ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
            break;
         case   1:
            tnlAssert( false, cerr << "blockSize should not be 1." << endl );
            break;
         default:
            tnlAssert( false, cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
            break;
      }
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;
   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

/***
 * The modified parallel reduction - version 5.
 * In comparison to the version 4 we start with the sequential reduction i.e.
 *  each thread reads more then two elements (as it was in the modification 4)
 *  and reduce them in sequential loop. This modification gives the greatest
 *  improvement of the performance.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction2
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel5( const int size,
                                               const T* deviceInput,
                                               T* deviceOutput,
                                               T* dbg_array1 = 0 )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id i this block.
    * gridSize is the number of element processed by all blocks at the
    * same time.
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
   unsigned int lastTId = size - 2 * blockIdx. x * blockDim. x;
   unsigned int gridSize = 2 * blockDim. x * gridDim.x;

   /***
    * Read data into the shared memory. We start with
    * sequential reduction. This modification gives the
    * greatest performance improvement :-).
    */
   if( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlSum )
         sdata[ tid ] = deviceInput[ gid ] + deviceInput[ gid + blockDim. x ];
   }
   else if( gid < size )
   {
      sdata[ tid ] = deviceInput[ gid ];
   }
   gid += gridSize;
   while( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], :: tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], :: tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlSum )
         sdata[ tid ] += deviceInput[gid] + deviceInput[ gid + blockDim. x ];
      gid += gridSize;
   }
   if( gid < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlSum )
         sdata[ tid ] += deviceInput[gid];
   }
   __syncthreads();

   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which is one half of the elements to reduce.
    *  Each thread with ID < s reduce elements tid and tid + s. The result is stored
    *  in shared memory in sdata 0 .. s. We set s = s / 2 ( i.e. s >>= 1) and repeat
    *  the algorithm again until s = 1.
    *  We also separate the case when the blockDim. x is power of 2 and the algorithm
    *  can be written in more efficient way without some conditions.
    */
   unsigned int n = lastTId < blockDim. x ? lastTId : blockDim. x;
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
   {
      for( unsigned int s = n / 2; s > 0; s >>= 1 )
      {
         reduceAligned< T, operation >( tid, s, sdata );
         __syncthreads();
      }
   }
   else
   {
      for( unsigned int s = n / 2; s > 0; s >>= 1 )
      {
         reduceNonAligned< T, operation >( tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
   }

   /***
    * Store the result back in the global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling modified CUDA kernel (version 5) for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 * WARNING: This template calls very inefficient kernel. Use just tnlCUDAReduction instead.
 */
template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction5( const int size,
                              const T* deviceInput,
                              T& result,
                              T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    * @param desGridSize is desired grid size
    */
   const int desBlockSize = 128;
   const int desGridSize = 2048;

   T* dbg_array1; // debuging array

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      gridSize. x = :: Min( ( int ) ( sizeReduced / blockSize. x + 1 ) / 2, desGridSize );
      //if( gridSize. x * 2 * blockSize. x < sizeReduced )
      //    gridSize. x ++;
      int shmem = blockSize. x * sizeof( T );
      /*cout << "Size: " << sizeReduced
           << " Grid size: " << gridSize. x
           << " Block size: " << blockSize. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel5< T, operation ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;

      // debuging part
      /*T* host_array = new T[ desBlockSize ];
      cudaMemcpy( host_array, dbg_array1,  desBlockSize * sizeof( T ), cudaMemcpyDeviceToHost );
      for( int i = 0; i< :: Min( ( int ) blockSize. x, desBlockSize ); i ++ )
          cout << host_array[ i ] << " ";
      cout << endl;

      T* output = new T[ sizeReduced ];
      cudaMemcpy( output, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < sizeReduced; i ++ )
          cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

/***
 * The modified parallel reduction - version 4.
 * In comparison to the version 3 we reduce the number of threads by one half
 * avoiding the threads which are doing nothing.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction2
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel4( const int size,
                                               const T* deviceInput,
	                                       T* deviceOutput,
	                   	               T* dbg_array1 = 0  )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id i this block
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
   unsigned int lastTId = size - 2 * blockIdx. x * blockDim. x;

   /***
    * Read data into the shared memory. Each block process
    * 2 * blockDim. x data elements. Therefore we now reduce
    * data elements with indecis gid and gid + blcokDim. x.
    * If there is no element with index gid + blockDim. x we just
    * read element with the index gid into the shared memory.
    */
   if( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlSum )
         sdata[ tid ] = deviceInput[ gid ] + deviceInput[ gid + blockDim. x ];
   }
   else if( gid < size )
   {
      sdata[ tid ] = deviceInput[ gid ];
   }
   __syncthreads();

   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which is one half of the elements to reduce.
    *  Each thread with ID < s reduce elements tid and tid + s. The result is stored
    *  in shared memory in sdata 0 .. s. We set s = s / 2 ( i.e. s >>= 1) and repeat
    *  the algorithm again until s = 1.
    *  We also separate the case when the blovkDim. x is power of 2 and the algorithm
    *  can be written in more efficient way without some conditions.
    */
   unsigned int n = lastTId < blockDim. x ? lastTId : blockDim. x;
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
   {
      for( unsigned int s = n / 2; s > 0; s >>= 1 )
      {
         if( tid < s )
         {
            if( operation == tnlMin )
               sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlMax )
               sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlSum )
               sdata[ tid ] += sdata[ tid + s ];
         }
         __syncthreads();
      }
   }
   else
   {
      for( unsigned int s = n / 2; s > 0; s >>= 1 )
      {
         if( tid < s )
         {
            if( operation == tnlMin )
               sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlMax )
               sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlSum )
               sdata[ tid ] += sdata[ tid + s ];
         }
         /***
          * This is for the case when we have odd number of elements.
          * The last one will be reduced using the thread with ID 0.
          */
         __syncthreads();
         if( 2 * s < n && tid == n - 1 )
         {
            if( operation == tnlMin )
               sdata[ 0 ] = tnlCudaMin( sdata[ 0 ], sdata[ tid ] );
            if( operation == tnlMax )
               sdata[ 0 ] = tnlCudaMax( sdata[ 0 ], sdata[ tid ] );
            if( operation == tnlSum )
               sdata[ 0 ] += sdata[ tid ];
         }
         n = s;
         __syncthreads();
      }
   }

   /***
    * Store the result back in the global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling modified CUDA kernel (version 4) for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 * WARNING: This template calls very inefficient kernel. Use just tnlCUDAReduction instead.
 */
template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction4( const int size,
	                          const T* deviceInput,
	                          T& result,
	                          T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    */
   const int desBlockSize = 128;

   T* dbg_array1; // debuging array

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      /***
       * If 2 * blockSize. x does not devide sizeReduced we must increase the grid size by one block.
       * Example: sizeReduced = 5, blockSize. x = 2 => gridSize. x = 5 / 2 / 2 = 1.
       *  Now we have one block with blockSize. x = 2 i.e. 2 threads each of which processes
       *  2 data elements and the last element is going to be omitted.
       */

      gridSize. x = sizeReduced / blockSize. x / 2;
      if( gridSize. x * 2 * blockSize. x < sizeReduced )
    	  gridSize. x ++;
      int shmem = blockSize. x * sizeof( T );
      /*cout << "Size: " << sizeReduced
           << " Grid size: " << gridSize. x
           << " Block size: " << blockSize. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel4< T, operation ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux, dbg_array1 );
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;

      // debuging part
      /*T* host_array = new T[ desBlockSize ];
      cudaMemcpy( host_array, dbg_array1,  desBlockSize * sizeof( T ), cudaMemcpyDeviceToHost );
      for( int i = 0; i< :: Min( ( int ) blockSize. x, desBlockSize ); i ++ )
    	  cout << host_array[ i ] << " ";
      cout << endl;

      T* output = new T[ sizeReduced ];
      cudaMemcpy( output, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < sizeReduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

/***
 * The modified but still very slow parallel reduction - version 3.
 * In comparison to the version 2 the first half of the threads read
 * the first half of the data for the reduction and reduce them with the second half of the data.
 * This improves the memory access to the global but also the shared memory.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction2
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel3( const int size,
                                               const T* deviceInput,
		                               T* deviceOutput )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id i this block
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = blockIdx. x * blockDim. x + threadIdx. x;
   unsigned int lastTId = size - blockIdx. x * blockDim. x;

   /***
    * Read data into the shared memory.
    */
   if( gid < size )
      sdata[ tid ] = deviceInput[gid];
   __syncthreads();

   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which is one half of the elements to reduce.
    *  Each thread with ID < s reduce elements tid and tid + s. The result is stored
    *  in shared memory in sdata 0 .. s. We set s = s / 2 ( i.e. s >>= 1) and repeat
    *  the algorithm again until s = 1.
    */
   if( lastTId <= blockDim. x )
   {
      unsigned int n = blockDim. x;
      for( unsigned int s = n / 2; s > 0; s >>= 1 )
      {
         if( tid < s && tid + s < lastTId )
         {
            if( operation == tnlMin )
               sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlMax )
               sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlSum )
               sdata[ tid ] += sdata[ tid + s ];
         }
         /***
          * This is for the case when we have odd number of elements.
          * The last one will be reduced using the thread with ID 0.
          */
         if( 2 * s < n && tid == 0 )
         {
            if( operation == tnlMin )
               sdata[ 0 ] = tnlCudaMin( sdata[ 0 ], sdata[ n - 1 ] );
            if( operation == tnlMax )
               sdata[ 0 ] = tnlCudaMax( sdata[ 0 ], sdata[ n - 1 ] );
            if( operation == tnlSum )
               sdata[ 0 ] += sdata[ n - 1 ];
         }
         n = s;
         __syncthreads();
      }
   }
   else
   {
      for( unsigned int s = blockDim. x / 2; s > 0; s >>= 1 )
      {
         if( tid < s )
         {
            if( operation == tnlMin )
               sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlMax )
               sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
            if( operation == tnlSum )
               sdata[ tid ] += sdata[ tid + s ];
         }
         __syncthreads();
      }
   }

   /***
    *  Store the result back in the global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling modified CUDA kernel (version 3) for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 * WARNING: This template calls very inefficient kernel. Use just tnlCUDAReduction instead.
 */
template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction3( const int size,
	                      const T* deviceInput,
	                      T& result,
	                      T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    */
   const int desBlockSize = 128;

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      /***
       * If blockSize. x does not devide sizeReduced we must increase the grid size by one block.
       * Example: sizeReduced = 3, blockSize. x = 2 => gridSize. x = 3 / 2 = 1. Now we have one block with
       * blockSize. x = 2 and the last element is going to be omitted.
       */
      gridSize. x = sizeReduced / blockSize. x + ( sizeReduced % blockSize. x != 0 );
      int shmem = blockSize. x * sizeof( T );
      /*cout << "Size: " << sizeReduced
           << " Grid size: " << gridSize. x
           << " Block size: " << blockSize. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel3< T, operation ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux );
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;
   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

/***
 * The modified but still very slow parallel reduction - version 2.
 * In comparison to the version 1 the data is not reduced by threads
 * with even thread id but by the first half of threads. It is more efficient mapping
 * of threads to the SIMD architecture. We have also replaced operation modulo in the reduction loop.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction2
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel2( const int size,
		                               const T* deviceInput,
		                               T* deviceOutput )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id i this block
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = blockIdx. x * blockDim. x + threadIdx. x;
   unsigned int lastTId = size - blockIdx. x * blockDim. x;

   /***
    * Read data into the shared memory.
    */
   if( gid < size )
      sdata[tid] = deviceInput[gid];
   __syncthreads();

   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which we double in each iteration.
    */
   if( lastTId <= blockDim. x )
   {
      for( unsigned int s = 1; s < blockDim. x; s *= 2 )
      {
         unsigned int inds = 2 * s * tid;
         if( inds < blockDim. x && inds + s < lastTId )
         {
            if( operation == tnlMin )
               sdata[ inds ] = tnlCudaMin( sdata[ inds ], sdata[ inds + s ] );
            if( operation == tnlMax )
               sdata[ inds ] = tnlCudaMax( sdata[ inds ], sdata[ inds + s ] );
            if( operation == tnlSum )
               sdata[ inds ] += sdata[ inds + s ];
         }
         __syncthreads();
      }
   }
   else
   {
      for( unsigned int s = 1; s < blockDim. x; s *= 2 )
      {
         unsigned int inds = 2 * s * tid;
         if( inds < blockDim. x )
         {
            if( operation == tnlMin )
               sdata[ inds ] = tnlCudaMin( sdata[ inds ], sdata[ inds + s ] );
            if( operation == tnlMax )
               sdata[ inds ] = tnlCudaMax( sdata[ inds ], sdata[ inds + s ] );
            if( operation == tnlSum )
               sdata[ inds ] += sdata[ inds + s ];
         }
         __syncthreads();
      }
   }

   /***
    *  Store the result back in global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling modified CUDA kernel (version 2) for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 * WARNING: This template calls very inefficient kernel. Use just tnlCUDAReduction instead.
 */
template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction2( const int size,
	                      const T* deviceInput,
	                      T& result,
	                      T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    */
   const int desBlockSize = 128;

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      /***
       * If blockSize. x does not devide sizeReduced we must increase the grid size by one block.
       * Example: sizeReduced = 3, blockSize. x = 2 => gridSize. x = 3 / 2 = 1. Now we have one block with
       * blockSize. x = 2 and the last element is going to be omitted.
       */
      gridSize. x = sizeReduced / blockSize. x + ( sizeReduced % blockSize. x != 0 );
      int shmem = blockSize. x * sizeof( T );
      /*cout << "Size: " << sizeReduced
           << " Grid size: " << gridSize. x
           << " Block size: " << blockSize. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel2< T, operation ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux );
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;

   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

/***
 * The simplest and very slow parallel reduction - version 1.
 * We run one thread for each element of data to be reduced.
 * One half of the threads (with even thread id) read the elements, reduce them and store
 * the result back (into shared memory). Thus we reduce the data to one half.
 * We repeat it until we have only one data element.
 *
 * WARNING: This kernel only reduce data in one block. Use rather tnlCUDASimpleReduction1
 *          to call this kernel then doing it by yourself.
 *          This kernel is very inefficient. It is here only for educative and testing reasons.
 *          Please use tnlCUDAReduction instead.
 *
 * The kernel parameters:
 * @param size is the number of all element to reduce - not just in one block.
 * @param deviceInput input data which we want to reduce
 * @param deviceOutput an array to which we write the result of reduction.
 *                     Each block of the grid writes one element in this array
 *                     (i.e. the size of this array equals the number of CUDA blocks).
 */
template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel1( const int size,
		                               const T* deviceInput,
		                               T* deviceOutput )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   T* sdata = reinterpret_cast< T* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id i this block
    */
   unsigned int tid = threadIdx. x;
   unsigned int gid = blockIdx. x * blockDim. x + threadIdx. x;
   int lastTId = size - blockIdx. x * blockDim. x;

   /***
    * Read data into the shared memory.
    */
   if( gid < size )
      sdata[tid] = deviceInput[gid];
   __syncthreads();

   /***
    *  Process the parallel reduction.
    *  We reduce the data with step s which we double in each iteration.
    */
   for( unsigned int s = 1; s < blockDim. x; s *= 2 )
   {
      if( ( tid % ( 2 * s ) ) == 0 && tid + s < lastTId )
      {
         T& a = sdata[ tid ];
         T& b = sdata[ tid + s ];
         if( operation == tnlMin )
            a = tnlCudaMin( a, b );
         if( operation == tnlMax )
            a = tnlCudaMax( a, b );
         if( operation == tnlSum )
            a += b;
      }
      __syncthreads();
   }

   /***
    *  Store the result back in global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

/***
 * The template calling the simplest CUDA kernel for vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlSum, tnlMin or tnlMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput is the pointer to an array storing the data we want
 *        to reduce. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 * WARNING: This template calls very inefficient kernel. Use just tnlCUDAReduction instead.
 */
template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction1( const int size,
	                      const T* deviceInput,
	                      T& result,
	                      T* deviceAux = 0 )
{
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    */
   const int desBlockSize = 128;

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   tnlLongVectorCUDA< T > deviceAuxVct( "tnlCUDAReduction:deviceAuxVct" );
   if( ! deviceAux )
   {
      int sizeAlloc = :: Max( 1, size / desBlockSize );
      if( ! deviceAuxVct. setNewSize( sizeAlloc ) )
         return false;
      deviceAux = deviceAuxVct. Data();
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   int sizeReduced = size;
   const T* reductionInput = deviceInput;
   while( sizeReduced > 1 )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      /***
       * If blockSize. x does not devide sizeReduced we must increase the grid size by one block.
       * Example: sizeReduced = 3, blockSize. x = 2 => gridSize. x = 3 / 2 = 1. Now we have one block with
       * blockSize. x = 2 and the last element is going to be omitted.
       */
      gridSize. x = sizeReduced / blockSize. x + ( sizeReduced % blockSize. x != 0 );
      int shmem = blockSize. x * sizeof( T );
      /*cout << "Size: " << sizeReduced
           << " Grid size: " << gridSize. x
           << " Block size: " << blockSize. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel1< T, operation ><<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput, deviceAux );
      sizeReduced = gridSize. x;
      reductionInput = deviceAux;
   }
   /***
    * We transfer reduced data from device to host.
    * If size equals 1 the previous loop was not processed and we read
    * data directly from the input.
    */
   if( size == 1 )
      cudaMemcpy( &result, deviceInput, sizeof( T ), cudaMemcpyDeviceToHost );
   else
      cudaMemcpy( &result, deviceAux, sizeReduced * sizeof( T ), cudaMemcpyDeviceToHost );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Unable to transfer reduced data from device to host." << endl;
      return false;
   }
   return true;
}

#endif /* HAVE_CUDA */

#endif /* TNLCUDAKERNELS_H_ */
