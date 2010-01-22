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
 * We also limit the grid size to 2048 - desGridSize.
 *
 */

template < class T, tnlOperation operation, int blockSize >
__global__ void tnlCUDAReductionKernel( const int size,
	         			                const int grid_size,
	                                    const T* d_input,
	                                    T* d_output,
	                                    T* dbg_array1 = 0  )
{
   extern __shared__ __align__ ( 8 ) T sdata[];

   // Read data into the shared memory
   int tid = threadIdx. x;
   int gid = 2 * blockDim. x * blockIdx. x + threadIdx. x;

   if( gid + blockSize < size )
   {
      if( operation == tnlMin ) sdata[ tid ] = :: tnlCudaMin( d_input[ gid ], d_input[ gid + blockSize ] );
      if( operation == tnlMax ) sdata[ tid ] = :: tnlCudaMax( d_input[ gid ], d_input[ gid + blockSize ] );
      if( operation == tnlSum ) sdata[ tid ] = d_input[ gid ] + d_input[ gid + blockSize ];
      //dbg_array1[ blockIdx. x * blockDim. x + threadIdx. x ] = -1;
   }
   else if( gid < size )
   {
      sdata[ tid ] = d_input[ gid ];
      //dbg_array1[ blockIdx. x * blockDim. x + threadIdx. x ] = -2;
   }
   gid += grid_size;

   while( gid + blockSize < size )
   {
      if( operation == tnlMin ) sdata[ tid ] = :: tnlCudaMin( sdata[ tid ], :: tnlCudaMin( d_input[ gid ], d_input[ gid + blockSize ] ) );
      if( operation == tnlMax ) sdata[ tid ] = :: tnlCudaMax( sdata[ tid ], :: tnlCudaMax( d_input[ gid ], d_input[ gid + blockSize ] ) );
      if( operation == tnlSum ) sdata[ tid ] += d_input[gid] + d_input[ gid + blockSize ];
      gid += grid_size;
      //dbg_array1[ blockIdx. x * blockDim. x + threadIdx. x ] = -4;
   }
   //dbg_array1[ blockIdx. x * blockDim. x + threadIdx. x ] = sdata[ tid ];
   __syncthreads();

   if( gid + blockDim. x < size )
   {
      if( operation == tnlMin )
         sdata[ tid ] = tnlCudaMin( d_input[ gid ], d_input[ gid + blockDim. x ] );
      if( operation == tnlMax )
         sdata[ tid ] = tnlCudaMax( d_input[ gid ], d_input[ gid + blockDim. x ] );
      if( operation == tnlSum )
         sdata[ tid ] = d_input[ gid ] + d_input[ gid + blockDim. x ];
   }
   else if( gid < size )
   {
      sdata[ tid ] = d_input[ gid ];
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

   // Store the result back in global memory
   if( tid == 0 )
      d_output[ blockIdx. x ] = sdata[ 0 ];
}

/*
 * CUDA reduction kernel caller.
 * block_size can be only some of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512.
 * d_input must reside on the device.
 */
template< class T, tnlOperation operation >
bool tnlCUDAReduction( const int size,
	               const T* device_input,
	               T& result,
	               T* device_output = 0 )
{
   if( ! size ) return false;


   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 16;    //Desired block size
   const int desGridSize = 2048;

   T* dbg_array1( 0 );

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;

       //cudaMalloc( ( void** ) &dbg_array1, size * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!
       //cudaMalloc( ( void** ) &dbg_array2, desBlockSize * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!!
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = :: Min( ( int ) ( size_reduced / block_size. x + 1 ) / 2, desGridSize );
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlAssert( shmem < 16384, cerr << shmem << " bytes are required." << endl; );
      tnlAssert( block_size. x <= 512, cerr << "Block size is " << block_size. x << endl; );
      switch( block_size. x )
      {
		  case 512:
			  tnlCUDAReductionKernel< T, operation, 512 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case 256:
			  tnlCUDAReductionKernel< T, operation, 256 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case 128:
			  tnlCUDAReductionKernel< T, operation, 128 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case  64:
			  tnlCUDAReductionKernel< T, operation,  64 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case  32:
			  tnlCUDAReductionKernel< T, operation,  32 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case  16:
			  tnlCUDAReductionKernel< T, operation,  16 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case   8:
			  tnlCUDAReductionKernel< T, operation,   8 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case   4:
			  tnlCUDAReductionKernel< T, operation,   4 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case   2:
			  tnlCUDAReductionKernel< T, operation,   2 ><<< grid_size, block_size, shmem >>>( size_reduced, 2 * grid_size. x * block_size. x, reduction_input, device_output, dbg_array1 );
			  break;
		  case   1:
			  tnlAssert( false, cerr << "blockSize should not be 1." << endl );
			  break;
		  default:
			  tnlAssert( false, cerr << "Block size is " << block_size. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
			  break;
      	}
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* host_array = new T[ size ];
      cudaMemcpy( host_array, dbg_array1,  size * sizeof( T ), cudaMemcpyDeviceToHost );
      for( int i = 0; i< size; i ++ )
    	  cout << host_array[ i ] << " - ";
      cout << endl;*/

      /*T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
   	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
	   cudaFree( device_output );
   return true;
}


/*
 * Modified parallel reduction - version 5.
 * We have replaced the for cycle with switch
 * and template parameter blockSize
 */

template < class T, tnlOperation operation, int blockSize >
__global__ void tnlCUDASimpleReductionKernel5( const int size,
	                                           const T* d_input,
	                                           T* d_output,
	                   	                       T* dbg_array1 = 0  )
{
	extern __shared__ T sdata[];

	// Read data into the shared memory
	int tid = threadIdx. x;
	int gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
	// Last thread ID which manipulates meaningful data
	//int last_tid = size - 2 * blockIdx. x * blockDim. x;
	if( gid + blockDim. x < size )
	{
		if( operation == tnlMin )
			sdata[ tid ] = tnlCudaMin( d_input[ gid ], d_input[ gid + blockDim. x ] );
		if( operation == tnlMax )
			sdata[ tid ] = tnlCudaMax( d_input[ gid ], d_input[ gid + blockDim. x ] );
		if( operation == tnlSum )
			sdata[ tid ] = d_input[ gid ] + d_input[ gid + blockDim. x ];
	}
	else if( gid < size )
	{
		sdata[ tid ] = d_input[ gid ];
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

	// Store the result back in global memory
	if( tid == 0 )
		d_output[ blockIdx. x ] = sdata[ 0 ];
}

template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction5( const int size,
	                          const T* device_input,
	                          T& result,
	                          T* device_output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 16;    //Desired block size

   //T* dbg_array1;

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;

       //cudaMalloc( ( void** ) &dbg_array1, desBlockSize * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!
       //cudaMalloc( ( void** ) &dbg_array2, desBlockSize * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!!
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = ( size_reduced / block_size. x + 1 ) / 2;
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlAssert( shmem < 16384, cerr << shmem << " bytes are required." );
      tnlAssert( shmem < 16384, cerr << shmem << " bytes are required." << endl; );
      switch( block_size. x )
      {
		  case 512:
			  tnlCUDASimpleReductionKernel5< T, operation, 512 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case 256:
			  tnlCUDASimpleReductionKernel5< T, operation, 256 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case 128:
			  tnlCUDASimpleReductionKernel5< T, operation, 128 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case  64:
			  tnlCUDASimpleReductionKernel5< T, operation,  64 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case  32:
			  tnlCUDASimpleReductionKernel5< T, operation,  32 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case  16:
			  tnlCUDASimpleReductionKernel5< T, operation,  16 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case   8:
			  tnlCUDASimpleReductionKernel5< T, operation,   8 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case   4:
			  tnlCUDASimpleReductionKernel5< T, operation,   4 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case   2:
			  tnlCUDASimpleReductionKernel5< T, operation,   2 ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output );
			  break;
		  case   1:
			  tnlAssert( false, cerr << "blockSize should not be 1." << endl );
			  break;
		  default:
			  tnlAssert( false, cerr << "Block size is " << block_size. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
			  break;
      	}
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* host_array = new T[ desBlockSize ];
      cudaMemcpy( host_array, dbg_array1,  desBlockSize * sizeof( T ), cudaMemcpyDeviceToHost );
      for( int i = 0; i< :: Min( ( int ) block_size. x, desBlockSize ); i ++ )
    	  cout << host_array[ i ] << " - ";
      cout << endl;

      T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
   	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
	   cudaFree( device_output );
   return true;
}


/*
 * Modified parallel reduction - version 4.
 * We have reduced the grid size to one half
 * to avoid inactive threads.
 */


template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel4( const int size,
	                                           const T* d_input,
	                                           T* d_output,
	                   	                       T* dbg_array1 = 0  )
{
	extern __shared__ T sdata[];

	// Read data into the shared memory
	int tid = threadIdx. x;
	int gid = 2 * blockIdx. x * blockDim. x + threadIdx. x;
	// Last thread ID which manipulates meaningful data
	int last_tid = size - 2 * blockIdx. x * blockDim. x;
	if( gid + blockDim. x < size )
	{
		if( operation == tnlMin )
			sdata[ tid ] = tnlCudaMin( d_input[ gid ], d_input[ gid + blockDim. x ] );
		if( operation == tnlMax )
			sdata[ tid ] = tnlCudaMax( d_input[ gid ], d_input[ gid + blockDim. x ] );
		if( operation == tnlSum )
			sdata[ tid ] = d_input[ gid ] + d_input[ gid + blockDim. x ];
	}
	else if( gid < size )
	{
		sdata[ tid ] = d_input[ gid ];
	}
	__syncthreads();
	//dbg_array1[ tid ] = tid; //sdata[ tid ];

	// Parallel reduction
	int n = last_tid < blockDim. x ? last_tid : blockDim. x;
	for( int s = n / 2; s > 0; s >>= 1 )
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
		if( 2 * s < n && tid == n - 1 )
		{
			if( operation == tnlMin )
				sdata[ 0 ] = tnlCudaMin( sdata[ 0 ], sdata[ tid ] );
			if( operation == tnlMax )
				sdata[ 0 ] = tnlCudaMax( sdata[ 0 ], sdata[ tid ] );
			if( operation == tnlSum )
				sdata[ 0 ] += sdata[ tid ];
			dbg_array1[ 0 ] = sdata[ tid ];
		}
		n = s;

		__syncthreads();
		//dbg_array1[ tid ] = -sdata[ tid ];

	}

	// Store the result back in the global memory
	if( tid == 0 )
		d_output[ blockIdx. x ] = sdata[ 0 ];
}

template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction4( const int size,
	                          const T* device_input,
	                          T& result,
	                          T* device_output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 16;    //Desired block size

   T* dbg_array1;

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;

       //cudaMalloc( ( void** ) &dbg_array1, desBlockSize * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!
       //cudaMalloc( ( void** ) &dbg_array2, desBlockSize * sizeof( T ) ); //!!!!!!!!!!!!!!!!!!!!!!!!!
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = size_reduced / block_size. x / 2;
      if( grid_size. x * 2 * block_size. x < size_reduced )
    	  grid_size. x ++;
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel4< T, operation ><<< grid_size. x, block_size, shmem >>>( size_reduced, reduction_input, device_output, dbg_array1 );
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* host_array = new T[ desBlockSize ];
      cudaMemcpy( host_array, dbg_array1,  desBlockSize * sizeof( T ), cudaMemcpyDeviceToHost );
      for( int i = 0; i< :: Min( ( int ) block_size. x, desBlockSize ); i ++ )
    	  cout << host_array[ i ] << " ";
      cout << endl;

      T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
	   cudaFree( device_output );
   return true;
}

/*
 * Modified parallel reduction - version 3.
 * We have avoided conflicting memory accesses.
 */

template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel3( const int size,
		                                       const T* d_input,
		                                       T* d_output )
{
	extern __shared__ T sdata[];

	// Read data into the shared memory
	int tid = threadIdx. x;
	int gid = blockIdx. x * blockDim. x + threadIdx. x;
	// Last thread ID which manipulates meaningful data
	int last_tid = size - blockIdx. x * blockDim. x;
	if( gid < size )
		sdata[ tid ] = d_input[gid];
	__syncthreads();

	// Parallel reduction
	int n = last_tid < blockDim. x ? last_tid : blockDim. x;
	for( int s = n / 2; s > 0; s >>= 1 )
	{
		if( tid < s && tid + s < last_tid )
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

	// Store the result back in global memory
	if( tid == 0 )
		d_output[ blockIdx. x ] = sdata[ 0 ];
}

template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction3( const int size,
	                          const T* device_input,
	                          T& result,
	                          T* device_output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 256;    //Desired block size

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = size_reduced / block_size. x + ( size_reduced % block_size. x != 0 );
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel3< T, operation ><<< grid_size, block_size, shmem >>>( size_reduced, reduction_input, device_output );
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
   	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
	   cudaFree( device_output );
   return true;
}

/*
 * Modified parallel reduction - version 2.
 * We have avoided operation modulo.
 */

template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel2( const int size,
		                                       const T* d_input,
		                                       T* d_output )
{
	extern __shared__ T sdata[];

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
		int inds = 2 * s * tid;
		if( inds < blockDim. x && inds + s < last_tid )
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

	// Store the result back in global memory
	if( tid == 0 )
		d_output[ blockIdx. x ] = sdata[ 0 ];
}

template< class T, tnlOperation operation >
bool tnlCUDASimpleReduction2( const int size,
	                          const T* device_input,
	                          T& result,
	                          T* device_output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 256;    //Desired block size

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = size_reduced / block_size. x + ( size_reduced % block_size. x != 0 );
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel2< T, operation ><<< grid_size, block_size, shmem >>>( size_reduced, reduction_input, device_output );
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
	   cudaFree( device_output );
   return true;
}

/*
 * The simplest and very slow parallel reduction - version 1.
 */

template < class T, tnlOperation operation >
__global__ void tnlCUDASimpleReductionKernel1( const int size,
		                                       const T* d_input,
		                                       T* d_output )
{
	extern __shared__ T sdata[];

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
bool tnlCUDASimpleReduction1( const int size,
	                          const T* device_input,
	                          T& result,
	                          T* device_output = 0 )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 256;    //Desired block size

   bool device_output_allocated( false );
   if( ! device_output )
   {
	   int bytes_alloc = :: Max( 1, size / desBlockSize ) * sizeof( T );
       cudaMalloc( ( void** ) &device_output, bytes_alloc );
       if( cudaGetLastError() != cudaSuccess )
       {
    	   cerr << "Unable to allocate device memory with size " << bytes_alloc << "." << endl;
    	   return false;
       }
       device_output_allocated = true;
   }
   dim3 block_size( 0 ), grid_size( 0 );
   int shmem;
   int size_reduced = size;
   const T* reduction_input = device_input;
   while( size_reduced > cpuThreshold )
   {
      block_size. x = :: Min( size_reduced, desBlockSize );
      grid_size. x = size_reduced / block_size. x + ( size_reduced % block_size. x != 0 );
      shmem = block_size. x * sizeof( T );
      /*cout << "Size: " << size_reduced
           << " Grid size: " << grid_size. x
           << " Block size: " << block_size. x
           << " Shmem: " << shmem << endl;*/
      tnlCUDASimpleReductionKernel1< T, operation ><<< grid_size, block_size, shmem >>>( size_reduced, reduction_input, device_output );
      size_reduced = grid_size. x;
      reduction_input = device_output;

      // debuging part
      /*T* output = new T[ size_reduced ];
      cudaMemcpy( output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
      cout << endl;
      for( int i = 0; i < size_reduced; i ++ )
    	  cout << output[ i ] << "   ";
      cout << endl;
      delete[] output;*/
   }
   T* host_output = new T[ size_reduced ];
   if( size == 1 )
	   cudaMemcpy( host_output, device_input, sizeof( T ), cudaMemcpyDeviceToHost );
   else
	   cudaMemcpy( host_output, device_output, size_reduced * sizeof( T ), cudaMemcpyDeviceToHost );
   result = host_output[ 0 ];
   for( int i = 1; i < size_reduced; i++ )
   {
      if( operation == tnlMin)
         result = :: Min( result, host_output[ i ] );
      if( operation == tnlMax )
         result = :: Max( result, host_output[ i ] );
      if( operation == tnlSum )
         result += host_output[ i ];
   }
   delete[] host_output;
   if( device_output_allocated )
   	   cudaFree( device_output );
   return true;
}

#endif /* HAVE_CUDA */

#endif /* TNLCUDAKERNELS_H_ */
