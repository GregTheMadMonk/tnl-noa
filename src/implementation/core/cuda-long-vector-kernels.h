/***************************************************************************
                          cuda-long-vector-kernels.h  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef CUDALONGVECTORKERNELS_H_
#define CUDALONGVECTORKERNELS_H_

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <iostream>
#include <core/tnlAssert.h>
#include <core/cuda/reduction-operations.h>
#include <implementation/core/memory-operations.h>

using namespace std;


/****
 * This constant says that arrays smaller than its value
 * are going to be reduced on CPU.
 */
const int maxGPUReductionDataSize = 256;

#ifdef HAVE_CUDA


/***
 * For each thread in block with thread ID smaller then s this function reduces
 * data elements with indecis tid and tid + s. Here we assume that for each
 * tid the tid + s element also exists i.e. we have even number of elements.
 */
template< typename Operation >
__device__ void reduceAligned( const Operation& operation,
                               typename Operation :: IndexType tid,
                               typename Operation :: IndexType  s,
                               typename Operation :: RealType* sdata )
{
   if( tid < s )
   {
      sdata[ tid ] = operation. commonReductionOnDevice( tid, tid + s, sdata );
      /*if( operation == tnlParallelReductionMin )
         sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlParallelReductionMax )
         sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlParallelReductionSum )
         sdata[ tid ] += sdata[ tid + s ];
      if( operation == tnlParallelReductionAbsMin )
         sdata[ tid ] = tnlCudaMin( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ tid ] = tnlCudaMax( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionLpNorm ||
          operation == tnlParallelReductionSdot )
         sdata[ tid ] = sdata[ tid ] + sdata[ tid + s ];*/
   }
}

/***
 * For each thread in block with thread ID smaller then s this function reduces
 * data elements with indices tid and tid + s. This is a modified version of
 * the previous algorithm. Thid one works even for odd number of elements but
 * it is a bit slower.
 */
template< typename Operation >
__device__ void reduceNonAligned( const Operation& operation,
                                  typename Operation :: IndexType tid,
                                  typename Operation :: IndexType s,
                                  typename Operation :: IndexType n,
                                  typename Operation :: RealType* sdata )
{
   if( tid < s )
   {
      sdata[ tid ] = operation. commonReductionOnDevice( tid, tid + s, sdata );
      /*if( operation == tnlParallelReductionMin )
         sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlParallelReductionMax )
         sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + s ] );
      if( operation == tnlParallelReductionSum )
         sdata[ tid ] += sdata[ tid + s ];
      if( operation == tnlParallelReductionAbsMin )
         sdata[ tid ] = tnlCudaMin( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ tid ] = tnlCudaMax( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionLpNorm ||
          operation == tnlParallelReductionSdot )
         sdata[ tid ] = sdata[ tid ] + sdata[ tid + s ];*/
   }
   /* This is for the case when we have odd number of elements.
    * The last one will be reduced using the thread with ID 0.
    */
   if( s > 32 )
      __syncthreads();
   if( 2 * s < n && tid == n - 1 )
   {
      sdata[ 0 ] = operation. commonReductionOnDevice( 0, tid, sdata );
      /*if( operation == tnlParallelReductionMin )
         sdata[ 0 ] = tnlCudaMin( sdata[ 0 ], sdata[ tid ] );
      if( operation == tnlParallelReductionMax )
         sdata[ 0 ] = tnlCudaMax( sdata[ 0 ], sdata[ tid ] );
      if( operation == tnlParallelReductionSum )
         sdata[ 0 ] += sdata[ tid ];
      if( operation == tnlParallelReductionAbsMin )
         sdata[ 0 ] = tnlCudaMin( tnlCudaAbs( sdata[ 0] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ 0 ] = tnlCudaMax( tnlCudaAbs( sdata[ 0 ] ), tnlCudaAbs( sdata[ tid + s ] ) );
      if( operation == tnlParallelReductionLpNorm ||
          operation == tnlParallelReductionSdot )
         sdata[ 0 ] = sdata[ 0 ] + sdata[ tid + s ];*/

   }
}

/***
 * The parallel reduction of one vector.
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
template < typename Operation, int blockSize >
__global__ void tnlCUDAReductionKernel( const Operation operation,
                                        const typename Operation :: IndexType size,
                                        const typename Operation :: RealType* deviceInput,
                                        const typename Operation :: RealType* deviceInput2,
                                        typename Operation :: RealType* deviceOutput )
{
   extern __shared__ __align__ ( 8 ) char __sdata[];
   
   typedef typename Operation :: IndexType IndexType;
   typedef typename Operation :: RealType RealType;
   RealType* sdata = reinterpret_cast< RealType* >( __sdata );

   /***
    * Get thread id (tid) and global thread id (gid).
    * lastTId is the last relevant thread id in this block.
    * gridSize is the number of element processed by all blocks at the
    * same time.
    */
   IndexType tid = threadIdx. x;
   IndexType gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
   IndexType lastTId = size - 2 * blockIdx. x * blockDim. x;
   IndexType gridSize = 2 * blockDim. x * gridDim.x;

   /***
    * Read data into the shared memory. We start with the
    * sequential reduction.
    */
   if( gid + blockDim. x < size )
   {
      sdata[ tid ] = operation. initialValueOnDevice( gid, gid + blockDim. x, deviceInput, deviceInput2 );
      /*if( operation == tnlParallelReductionMin )
         sdata[ tid ] = tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlParallelReductionMax )
         sdata[ tid ] = tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] );
      if( operation == tnlParallelReductionAbsMin )
         sdata[ tid ] = tnlCudaMin( tnlCudaAbs( deviceInput[ gid ] ), tnlCudaAbs( deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ tid ] = tnlCudaMax( tnlCudaAbs( deviceInput[ gid ] ), tnlCudaAbs( deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlParallelReductionSum )
         sdata[ tid ] = deviceInput[ gid ] + deviceInput[ gid + blockDim. x ];
      if( operation == tnlParallelReductionLpNorm )
         sdata[ tid ] = powf( tnlCudaAbs( deviceInput[ gid ] ), parameter ) +
                        powf( tnlCudaAbs( deviceInput[ gid + blockDim. x ] ), parameter );
      if( operation == tnlParallelReductionSdot )
         sdata[ tid ] = deviceInput[ gid ] * deviceInput2[ gid ] +
                        deviceInput[ gid + blockDim. x ] * deviceInput2[ gid + blockDim. x ];*/
   }
   else if( gid < size )
   {
      sdata[ tid ] = operation. initialValueOnDevice( gid, deviceInput, deviceInput2 );
      /*if( operation == tnlParallelReductionLpNorm )
         sdata[ tid ] = powf( tnlCudaAbs( deviceInput[ gid ] ), parameter );
      else
         if( operation == tnlParallelReductionSdot )
            sdata[ tid ] = deviceInput[ gid ] * deviceInput2[ gid ];
         else
            sdata[ tid ] = deviceInput[ gid ];*/
   }
   gid += gridSize;
   while( gid + blockDim. x < size )
   {
      sdata[ tid ] = operation. firstReductionOnDevice( tid, gid, gid + blockDim. x, sdata, deviceInput, deviceInput2 );
      /*if( operation == tnlParallelReductionMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], :: tnlCudaMin( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlParallelReductionMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], :: tnlCudaMax( deviceInput[ gid ], deviceInput[ gid + blockDim. x ] ) );
      if( operation == tnlParallelReductionAbsMin )
         sdata[ tid ] = :: tnlCudaMin( tnlCudaAbs( sdata[ tid ] ), :: tnlCudaMin( tnlCudaAbs( deviceInput[ gid ] ), tnlCudaAbs( deviceInput[ gid + blockDim. x ] ) ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ tid ] = :: tnlCudaMax( tnlCudaAbs( sdata[ tid ] ), :: tnlCudaMax( tnlCudaAbs( deviceInput[ gid ] ), tnlCudaAbs( deviceInput[ gid + blockDim. x ] ) ) );
      if( operation == tnlParallelReductionSum )
         sdata[ tid ] += deviceInput[gid] + deviceInput[ gid + blockDim. x ];
      if( operation == tnlParallelReductionLpNorm )
         sdata[ tid ] += powf( tnlCudaAbs( deviceInput[gid] ), parameter ) +
                         powf( tnlCudaAbs( deviceInput[ gid + blockDim. x ] ), parameter );
      if( operation == tnlParallelReductionSdot )
         sdata[ tid ] += deviceInput[ gid ] * deviceInput2[ gid ] +
                         deviceInput[ gid + blockDim. x] * deviceInput2[ gid + blockDim. x ];*/
      gid += gridSize;
   }
   if( gid < size )
   {
      sdata[ tid ] = operation. firstReductionOnDevice( tid, gid, sdata, deviceInput, deviceInput2 );
      /*if( operation == tnlParallelReductionMin )
         sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlParallelReductionMax )
         sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], deviceInput[ gid ] );
      if( operation == tnlParallelReductionAbsMin )
         sdata[ tid ] = :: tnlCudaMin( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( deviceInput[ gid ] ) );
      if( operation == tnlParallelReductionAbsMax )
         sdata[ tid ] = :: tnlCudaMax( tnlCudaAbs( sdata[ tid ] ), tnlCudaAbs( deviceInput[ gid ] ) );
      if( operation == tnlParallelReductionSum )
         sdata[ tid ] += deviceInput[gid];
      if( operation == tnlParallelReductionLpNorm )
         sdata[ tid ] += powf( tnlCudaAbs( deviceInput[gid] ), parameter );
      if( operation == tnlParallelReductionSdot )
         sdata[ tid ] += deviceInput[ gid ] * deviceInput2[ gid ];*/
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
      if( blockSize >= 512 )
      {
         if( tid < 256 )
            reduceAligned( operation, tid, 256, sdata );
         __syncthreads();
      }
      if( blockSize >= 256 )
      {
         if( tid < 128 )
            reduceAligned( operation, tid, 128, sdata );
         __syncthreads();
      }
      if( blockSize >= 128 )
      {
         if( tid <  64 )
            reduceAligned( operation, tid, 64, sdata );
         __syncthreads();
      }

      /***
       * This runs in one warp so it is synchronised implicitly.
       */
      if (tid < 32)
      {
         if( blockSize >= 64 )
            reduceAligned( operation, tid, 32, sdata );
         if( blockSize >= 32 )
            reduceAligned( operation, tid, 16, sdata );
         if( blockSize >= 16 )
            reduceAligned( operation, tid,  8, sdata );
         if( blockSize >=  8 )
            reduceAligned( operation, tid,  4, sdata );
         if( blockSize >=  4 )
            reduceAligned( operation, tid,  2, sdata );
         if( blockSize >=  2 )
            reduceAligned( operation, tid,  1, sdata );
      }
   }
   else
   {
      unsigned int s;
      if( n >= 512 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 256 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 128 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 64 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      if( n >= 32 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
         __syncthreads();
      }
      /***
       * This runs in one warp so it is synchronised implicitly.
       */
      if( n >= 16 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
      }
      if( n >= 8 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
      }
      if( n >= 4 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
      }
      if( n >= 2 )
      {
         s = n / 2;
         reduceNonAligned( operation, tid, s, n, sdata );
         n = s;
      }

   }

   /***
    * Store the result back in the global memory.
    */
   if( tid == 0 )
      deviceOutput[ blockIdx. x ] = sdata[ 0 ];
}

#endif
/***
 * The template calling the final CUDA kernel for the single vector reduction.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlParallelReductionSum, tnlParallelReductionMin or tnlParallelReductionMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput1 is the pointer to an array storing the data we want
 *        to reduce. This array must stay on the device!.
 * @param deviceInput2 is the pointer to an array storing the coupling data for example
 *        the second vector for the SDOT operation. This array must stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param parameter can be used for example for the passing the parameter p of Lp norm.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 */
template< typename Operation >
bool tnlCUDALongVectorReduction( const Operation& operation,
                                 const typename Operation :: IndexType size,
                                 const typename Operation :: RealType* deviceInput1,
                                 const typename Operation :: RealType* deviceInput2,
                                 typename Operation :: RealType& result,
                                 typename Operation :: RealType* deviceAux = 0 )
{
#ifdef HAVE_CUDA

   typedef typename Operation :: IndexType IndexType;
   typedef typename Operation :: RealType RealType;
   /***
    * Set parameters:
    * @param desBlockSize is desired block size with which we get the best performance (on CUDA rach 1.0 to 1.3)
    * @param desGridSize is desired grid size
    */
   const int desBlockSize = 512;
   const int desGridSize = 2048;

   RealType* dbg_array1; // debuging array

   /***
    * Allocating auxiliary device memory to store temporary reduced arrays.
    * For example in the first iteration we reduce the number of elements
    * from size to size / 2. We store this new data in deviceAux array.
    * If one calls the CUDA reduction more then once then one can provide
    * auxiliary array by passing it via the parameter deviceAux.
    */
   RealType* myDeviceAux( 0 );
   if( ! deviceAux )
   {
      if( ! allocateMemoryCuda( myDeviceAux, :: Max( 1, size / desBlockSize ) ) )
      {
         checkCudaDevice;
         return false;
      }
      deviceAux = myDeviceAux;
   }

   /***
    * Setup parameters of the kernel:
    * @param sizeReduced is the size of reduced data after each step of parallel reduction
    * @param reductionInput tells what data we shell reduce. We start with the input if this fuction
    *                       and after the 1st reduction step we switch this pointer to deviceAux.
    */
   IndexType sizeReduced = size;
   const RealType* reductionInput1 = deviceInput1;
   const RealType* reductionInput2 = deviceInput2;
   IndexType reductionSteps( 0 );
   while( sizeReduced > maxGPUReductionDataSize )
   {
      dim3 blockSize( 0 ), gridSize( 0 );
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      gridSize. x = :: Min( ( IndexType ) ( sizeReduced / blockSize. x + 1 ) / 2, desGridSize );

      /***
       * We align the blockSize to the power of 2.
       */
      IndexType alignedBlockSize = 1;
      while( alignedBlockSize < blockSize. x ) alignedBlockSize <<= 1;
      blockSize. x = alignedBlockSize;
      IndexType shmem = blockSize. x * sizeof( RealType );
      /***
       * Depending on the blockSize we generate appropriate template instance.
       */
#ifdef UNDEF
      if( reductionSteps > 0 &&
          ( operation == tnlParallelReductionSdot ||
            operation == tnlParallelReductionLpNorm ) )
      {
         /***
          * For operations like SDOT or LpNorm we need to switch to tnlParallelReductionSum after the
          * first reduction step.
          */
         switch( blockSize. x )
         {
            case 512:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum, 512 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case 256:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum, 256 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case 128:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum, 128 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case  64:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,  64 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case  32:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,  32 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case  16:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,  16 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case   8:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,   8 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case   4:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,   4 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case   2:
               tnlCUDAReductionKernel< Type, ParameterType, Index, tnlParallelReductionSum,   2 >
               <<< gridSize, blockSize, shmem >>>( sizeReduced, reductionInput1, reductionInput2, deviceAux, parameter, dbg_array1 );
               break;
            case   1:
               tnlAssert( false, cerr << "blockSize should not be 1." << endl );
               break;
            default:
               tnlAssert( false, cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
               break;
         }
      }
      else
#endif
         switch( blockSize. x )
         {
            case 512:
               tnlCUDAReductionKernel< Operation, 512 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case 256:
               tnlCUDAReductionKernel< Operation, 256 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case 128:
               tnlCUDAReductionKernel< Operation, 128 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case  64:
               tnlCUDAReductionKernel< Operation,  64 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case  32:
               tnlCUDAReductionKernel< Operation,  32 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case  16:
               tnlCUDAReductionKernel< Operation,  16 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case   8:
               tnlCUDAReductionKernel< Operation,   8 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case   4:
               tnlCUDAReductionKernel< Operation,   4 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case   2:
               tnlCUDAReductionKernel< Operation,   2 >
               <<< gridSize, blockSize, shmem >>>( operation, sizeReduced, reductionInput1, reductionInput2, deviceAux );
               break;
            case   1:
               tnlAssert( false, cerr << "blockSize should not be 1." << endl );
               break;
            default:
               tnlAssert( false, cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
               break;
         }
      sizeReduced = gridSize. x;
      reductionInput1 = deviceAux;
      reductionSteps ++;
   }

   /***
    * We transfer reduced data from device to host.
    * If sizeReduced equals size the previous loop was not processed and we read
    * data directly from the input.
    */
   RealType result_array[ maxGPUReductionDataSize ];
   RealType result_array2[ maxGPUReductionDataSize ];
   if( sizeReduced == size )
   {
      if( ! copyMemoryCudaToHost( result_array, deviceInput1, sizeReduced ) )
         return false;
#ifdef UNDEF
      switch( operation )
      {
         case tnlParallelReductionLpNorm:
            result = pow( tnlAbs( result_array[ 0 ] ), parameter );
            for( Index i = 1; i < sizeReduced; i ++ )
               result += pow( tnlAbs( result_array[ i ] ), parameter );
            result = pow( result, 1.0/ parameter );
            return true;
         case tnlParallelReductionSdot:
            if( ! copyMemoryCudaToHost( result_array2, deviceInput2, sizeReduced ) )
               return false;
            else
            {
               result = 0;
               for( Index i = 0; i < sizeReduced; i ++ )
                  result += result_array[ i ] * result_array2[ i ] ;
               return true;
            }
      }
#endif
   }
   else
      if( ! copyMemoryCudaToHost( result_array, deviceAux, sizeReduced ) )
         return false;
   result = result_array[ 0 ];
   for( IndexType i = 1; i < sizeReduced; i ++ )
      result = operation. reduceOnHost( result, result_array[ i ] );
   /*switch( operation )
   {
      case tnlParallelReductionMax:
         result = result_array[ 0 ];
         for( Index i = 1; i < sizeReduced; i ++ )
            result = Max( result, result_array[ i ] );
         break;
      case tnlParallelReductionMin:
         result = result_array[ 0 ];
         for( Index i = 1; i < sizeReduced; i ++ )
            result = Min( result, result_array[ i ] );
         break;
      case tnlParallelReductionSum:
      case tnlParallelReductionLpNorm:
      case tnlParallelReductionSdot:
         result = result_array[ 0 ];
         for( Index i = 1; i < sizeReduced; i ++ )
            result += result_array[ i ];
         break;
      case tnlParallelReductionAbsMax:
         result = tnlAbs( result_array[ 0 ] );
         for( Index i = 1; i < sizeReduced; i ++ )
            result = Max( result, tnlAbs( result_array[ i ] ) );
         break;
      case tnlParallelReductionAbsMin:
         result = tnlAbs( result_array[ 0 ] );
         for( Index i = 1; i < sizeReduced; i ++ )
            result = Min( result, tnlAbs( result_array[ i ] ) );
         break;
   }*/
   if( myDeviceAux )
   {
      freeMemoryCuda( myDeviceAux );
      if( ! checkCudaDevice )
         return false;
   }
   return true;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
};

#ifdef HAVE_CUDA
/***
 * This kernel just compares two vectors element by element. It writes
 * the result of the comparison into array result. This array must be
 * then reduced.
 */
template< typename Real, typename Index >
__global__ void compareTwoVectorsElementwise( const Index size,
                                              const Real* vector1,
                                              const Real* vector2,
                                              bool* result )
{
   Index gid = blockDim. x * blockIdx. x + threadIdx. x;
   if( gid < size )
   {
      if( vector1[ gid ] == vector2[ gid ] )
         result[ gid ] = true;
      else
         result[ gid ] = false;
   }
}
#endif

/***
 * The template for comparison of two long vectors on the CUDA device.
 * The template parameters are:
 * @param T is the type of data we want to reduce
 * @param operation is the operation reducing the data.
 *        It can be tnlParallelReductionSum, tnlParallelReductionMin or tnlParallelReductionMax.
 * The function parameters:
 * @param size tells number of elements in the data array.
 * @param deviceInput1 is the pointer to an array storing the data we want
 *        to reduce. This array must stay on the device!.
 * @param deviceInput2 is the pointer to an array storing the coupling data for example
 *        the second vector for the SDOT operation. This array most stay on the device!.
 * @param result will contain the result of the reduction if everything was ok
 *        and the return code is true.
 * @param deviceAux is auxiliary array used to store temporary data during the reduction.
 *        If one calls this function more then once one might provide this array to avoid repetetive
 *        allocation of this array on the device inside of this function.
 *        The size of this array should be size / 128 * sizeof( T ).
 *
 * This function first calls kernel which compares each couples of elements from both vectors.
 * Result is written into a bool array. The minimum value then says if both vectors equal.
 *
 */
// TODO: result of comparison should not be returned!!!
template< typename Type, typename Index >
bool tnlCUDALongVectorComparison( const Index size,
                                  const Type* deviceInput1,
                                  const Type* deviceInput2,
                                  bool* deviceBoolAux = 0,
                                  Type* deviceAux = 0 )
{
#ifdef HAVE_CUDA
   tnlAssert( size > 0,
              cerr << "You try to compare two CUDA long vectors with non-positive size." << endl
                   << "The size is " << size );
   //tnlVector< bool, tnlCuda, Index > boolArray( "tnlCUDALongVectorComparison:bool_array" );
   bool* myDeviceBoolAux( 0 );
   if( ! deviceBoolAux )
   {
      //if( ! boolArray. setSize( size ) )
      if( ! allocateMemoryCuda( myDeviceBoolAux, size ) )
         return false;
      deviceBoolAux = myDeviceBoolAux;
   }
   dim3 blockSize( 0 ), gridSize( 0 );
   blockSize. x = 256;
   gridSize. x = size / blockSize. x + 1;

   compareTwoVectorsElementwise<<< gridSize, blockSize >>>( size,
                                                            deviceInput1,
                                                            deviceInput2,
                                                            deviceBoolAux );
   if( ! checkCudaDevice )
      return false;
   bool result;
   if( ! tnlCUDALongVectorReduction< bool, bool, Index, tnlParallelReductionMin >( size,
                                                                                   deviceBoolAux,
                                                                                   ( bool* ) NULL,
                                                                                   result,
                                                                                   0 ) )


      return false;
   return result;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
}

#endif /* CUDALONGVECTORKERNELS_H_ */
