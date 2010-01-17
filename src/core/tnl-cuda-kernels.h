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

using namespace std;

enum tnlOperation { tnlMin, tnlMax, tnlSum };

#ifdef HAVE_CUDA

template< class T > __device__ T tnlCudaMin( const T& a,
		                                     const T& b )
{
	return a < b ? a : b;
}

template< class T > __device__ T tnlCudaMax( const T& a,
		                                     const T& b )
{
	return a > b ? a : b;
}

/*
 * This kernel has been adopted from the diploma work of Jan Vacata.
 * Vacata Jan, GPGPU: General Purpose Computation on GPUs, diploma thesis,
 *  Department of mathematics, FNSPE, CTU in Prague, 2008.
 *
 * Call this kernel with grid size divided by 2.
 * Maximum block size is 512.
 *
 */

template < class T, tnlOperation operation, int blockSize >
__global__ void tnlCUDAReductionKernel( const int size,
		                                const T* d_input,
		                                T* d_output )
{
	extern __shared__ __align__( 8 ) T sdata[];
	// Read the data into shared memory
	int tid = threadIdx.x;
	int gid = blockIdx. x * blockSize * 2 + threadIdx. x;
	int gridSize = blockSize * 2 * gridDim. x;
	if( operation == tnlMin ||
		operation == tnlMax )
		sdata[ tid ] = d_input[ gid ];
	else
		sdata[ tid ] = 0;
	while( gid < size )
	{
		if( operation == tnlMin )
			sdata[ tid ] = tnlCudaMin( d_input[ gid ], d_input[ tnlCudaMin( gid + blockSize, size ) ] );
		if( operation == tnlMax )
			sdata[ tid ] = tnlCudaMax( d_input[ gid ], d_input[ tnlCudaMin( gid + blockSize, size ) ] );
		if( operation == tnlSum )
			sdata[ tid ] += d_input[gid] + d_input[gid+blockSize];
		gid += gridSize;
	}
	__syncthreads();

	// Parallel reduction
	if( blockSize == 512 )
	{
		if( tid < 256 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 256 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 256 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 256 ];
		}
		__syncthreads();
	}
	if( blockSize >= 256 )
	{
		if( tid < 128 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 128 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 128 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 128 ];
		}
		__syncthreads();
	}
	if( blockSize >= 128 )
	{
		if (tid< 64)
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 64 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 64 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 64 ];
		}
		__syncthreads();
	}
	/*
	 * What follows runs in warp so it does not need to be synchronised.
	 */
	if( tid < 32 )
	{
		if( blockSize >= 64 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 32 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 32 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 32 ];
		}
		if( blockSize >= 32 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 16 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 16 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 16 ];
		}
		if( blockSize >= 16 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 8 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 8 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 8 ];
		}
	    if( blockSize >= 8 )
	    {
	    	if( operation == tnlMin )
	    		sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 4 ] );
	    	if( operation == tnlMax )
	    		sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 4 ] );
	    	if( operation == tnlSum )
	    		sdata[ tid ] += sdata[ tid + 4 ];
	    }
		if( blockSize >= 4 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 2 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 2 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 2 ];
		}
		if( blockSize >= 2 )
		{
			if( operation == tnlMin )
				sdata[ tid ] = tnlCudaMin( sdata[ tid ], sdata[ tid + 1 ] );
			if( operation == tnlMax )
				sdata[ tid ] = tnlCudaMax( sdata[ tid ], sdata[ tid + 1 ] );
			if( operation == tnlSum )
				sdata[ tid ] += sdata[ tid + 1 ];
		}
	}
	// Store the result back to the global memory of the device
	if( tid == 0 ) d_output[ blockIdx. x ] = sdata[ 0 ];
}

/*
 * CUDA reduction kernel caller.
 * block_size can be only some of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512.
 * d_input must reside on the device.
 */
template< class T, tnlOperation operation >
T tnlCUDAReduction( const int size,
					const int block_size,
					const int grid_size,
	                const T* d_input )
{
	T result;

	dim3 blockSize( block_size );
	dim3 gridSize( grid_size );
	int shmem = 512 * sizeof( T );
	tnlAssert( shmem < 16384, cerr << shmem << " bytes are required." );
	switch( block_size )
	{
		case 512:
	        tnlCUDAReductionKernel< T, operation, 512 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	        break;
	    case 256:
	    	tnlCUDAReductionKernel< T, operation, 256 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case 128:
	    	tnlCUDAReductionKernel< T, operation, 128 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case  64:
	    	tnlCUDAReductionKernel< T, operation,  64 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case  32:
	    	tnlCUDAReductionKernel< T, operation,  32 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case  16:
	    	tnlCUDAReductionKernel< T, operation,  16 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case   8:
	    	tnlCUDAReductionKernel< T, operation,   8 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case   4:
	    	tnlCUDAReductionKernel< T, operation,   4 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case   2:
	    	tnlCUDAReductionKernel< T, operation,   2 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    case   1:
	    	tnlCUDAReductionKernel< T, operation,   1 ><<< gridSize, blockSize, shmem >>>( size, d_input, &result );
	    	break;
	    default:
	    	tnlAssert( false, cerr << "Block size is " << block_size << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
	    	break;
	}
	return result;
}

template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel1( const int size,
		                                       const T* d_input,
		                                       T* d_output )
{
	extern __shared__ int sdata[];

	// Read data into the shared memory
	int tid = threadIdx. x;
	int gid = blockIdx. x * blockDim. x + threadIdx. x;
	// Last thread ID which manipulates meaningful data
	int last_tid = size - blockIdx. x * blockDim. x;
	if( gid < size )
		sdata[tid] = d_input[gid];
	__syncthreads();

	// Parallel reduction
	for( int s = 1; s < blockDim. x; s *= 2 )
	{
		if( ( tid % ( 2 * s ) ) == 0 && tid + s < last_tid )
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

	// Store the result back in global memory
	if( tid == 0 )
		d_output[ blockIdx. x ] = sdata[ 0 ];
}

template< class T, tnlOperation operation >
T tnlCUDASimpleReduction1( const int size,
	                   const T* d_input,
	                   T* output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 128;    //Desired block size
   dim3 block_size = :: Min( size, desBlockSize );
   dim3 grid_size = size / blockSize. x;
   unsigned int shmem = blockSize. x * sizeof( int );
   cout << "Grid size: " << grid_size. x << endl
        << "Block size: " << block_size. x << endl
        << "Shmem: " << shmem << endl;
   if( ! output )
   {
      cudaMalloc( ( void** ) &output, grid_size * sizeof( T ) );
      if( cuda // TODO: add allocation check
   }
   tnlCUDASimpleReductionKernel1< T, operation ><<< grid_size, block_size, shmem >>>( size, input, output );
   int size_reduced = grid_size. x;
   while( sizeReduced > cpuThreshold )
   {
      cout << "Reducing with size reduced = " << size_reduced << endl;
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = size_reduced / block_size. x;
      shmem = block_size. x * sizeof(int);
      tnlCUDASimpleReductionKernel1< T, operation ><<< grid_size, block_size, shmem >>>( size, input, output );
      size_reduced = grid_size. x;
   }
   int* host_output = new int[ size_reduced ];
   cudaMemcpy( host_output, output, sizeReduced * sizeof(int), cudaMemcpyDeviceToHost );
   int result = host_output[ 0 ];
   for( int i = 1;i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_ouput[ i ];
   }
   delete[] host_output;

   return result;
}

#endif /* HAVE_CUDA */

#endif /* TNLCUDAKERNELS_H_ */
