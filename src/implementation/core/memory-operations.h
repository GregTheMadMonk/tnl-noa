/***************************************************************************
                          memory-operations.h  -  description
                             -------------------
    begin                : Nov 9, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef MEMORYFUNCTIONS_H_
#define MEMORYFUNCTIONS_H_

#include <core/cuda/device-check.h>
#include <core/mfuncs.h>
#include <tnlConfig.h>

const int tnlGPUvsCPUTransferBufferSize( 1 << 20 );

template< typename Element, typename Index >
bool allocateMemoryHost( Element*& data,
                         const Index size )
{
   if( ! ( data = new Element[ size ] ) )
      return false;
   return true;
}

template< typename Element, typename Index >
bool allocateMemoryCuda( Element*& data,
                         const Index size )
{
#ifdef HAVE_CUDA
   if( cudaMalloc( ( void** ) &data,
                   ( size_t ) size * sizeof( Element ) ) != cudaSuccess )
      data = 0;
   return checkCudaDevice;
#endif
}

template< typename Element >
bool freeMemoryHost( Element* data )
{
   delete[] data;
   return true;
}

template< typename Element >
bool freeMemoryCuda( Element* data )
{
#ifdef HAVE_CUDA
      cudaFree( data );
      return checkCudaDevice;
#endif
   return true;
}

template< typename Element, typename Index >
bool setMemoryHost( Element* data,
                    const Element& value,
                    const Index size )
{
   for( Index i = 0; i < size; i ++ )
      data[ i ] = value;
   return true;
}

#ifdef HAVE_CUDA
template< typename Element, typename Index >
__global__ void setVectorValueCudaKernel( Element* data,
                                          const Index size,
                                          const Element value,
                                          const Index elementsPerThread )
{
   Index elementIdx = blockDim. x * blockIdx. x * elementsPerThread + threadIdx. x;
   Index elementsProcessed( 0 );
   while( elementsProcessed < elementsPerThread &&
          elementIdx < size )
   {
      data[ elementIdx ] = value;
      elementIdx += blockDim. x;
      elementsProcessed ++;
   }
}
#endif

template< typename Element, typename Index >
bool setMemoryCuda( Element* data,
                    const Element& value,
                    const Index size )
{
#ifdef HAVE_CUDA
      dim3 blockSize, gridSize;
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      Index elementsPerThread = ceil( ( double ) blocksNumber / ( double ) maxCudaGridSize );
      gridSize. x = Min( blocksNumber, maxCudaGridSize );
      //cout << "blocksNumber = " << blocksNumber << "Grid size = " << gridSize. x << " elementsPerThread = " << elementsPerThread << endl;
      setVectorValueCudaKernel<<< blockSize, gridSize >>>( data, size, value, elementsPerThread );

      return checkCudaDevice;
#else
      cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
      return false;
#endif

}

template< typename Element, typename Index >
bool copyMemoryHostToHost( Element* destination,
                           const Element* source,
                           const Index size )
{
   for( Index i = 0; i < size; i ++ )
      destination[ i ] = source[ i ];
   return true;
}

template< typename Element, typename Index >
bool copyMemoryHostToCuda( Element* destination,
                           const Element* source,
                           const Index size )
{
#ifdef HAVE_CUDA
   cudaMemcpy( destination,
               source,
               size * sizeof( Element ),
               cudaMemcpyHostToDevice );
   if( ! checkCudaDevice )
   {
      cerr << "Transfer of data from host to CUDA device failed." << endl;
      return false;
   }
   return true;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return false;
#endif
}


template< typename Element, typename Index >
bool copyMemoryCudaToHost( Element* destination,
                           const Element* source,
                           const Index size )
{
#ifdef HAVE_CUDA
   cudaMemcpy( destination,
               source,
               size * sizeof( Element ),
               cudaMemcpyDeviceToHost );
   if( ! checkCudaDevice )
   {
      cerr << "Transfer of data from CUDA device to host failed." << endl;
      return false;
   }
   return true;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return false;
#endif
}

template< typename Element, typename Index >
bool copyMemoryCudaToCuda( Element* destination,
                           const Element* source,
                           const Index size )
{
#ifdef HAVE_CUDA
   if( cudaMemcpy( destination,
                   source,
                   size * sizeof( Element ),
                   cudaMemcpyDeviceToDevice ) != cudaSuccess )
   return checkCudaDevice;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return false;
#endif
}

template< typename Element, typename Index >
bool compareMemoryHost( const Element* data1,
                        const Element* data2,
                        const Index size )
{
   for( Index i = 0; i < size; i ++ )
      if( data1[ i ] != data2[ i ] )
         return false;
   return true;
}

template< typename Element, typename Index >
bool compareMemoryHostCuda( const Element* hostData,
                               const Element* deviceData,
                               const Index size )
{
#ifdef HAVE_CUDA
   Index host_buffer_size = :: Min( ( Index ) ( tnlGPUvsCPUTransferBufferSize / sizeof( Element ) ),
                                                size );
   Element* host_buffer = new Element[ host_buffer_size ];
   if( ! host_buffer )
   {
      cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << endl;
      return false;
   }
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = Min( size - compared, host_buffer_size );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( deviceData[ compared ] ),
                      transfer * sizeof( Element ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         cerr << "Transfer of data from the device failed." << endl;
         checkCudaDevice;
         delete[] host_buffer;
         return false;
      }
      Index bufferIndex( 0 );
      while( bufferIndex < transfer &&
             host_buffer[ bufferIndex ] == hostData[ compared ] )
      {
         bufferIndex ++;
         compared ++;
      }
      if( bufferIndex < transfer )
      {
         delete[] host_buffer;
         return false;
      }
   }
   delete[] host_buffer;
   return true;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
}

template< typename Element, typename Index >
bool compareMemoryCuda( const Element* deviceData1,
                           const Element* deviceData2,
                           const Index size )
{
#ifdef HAVE_CUDA
   return tnlCUDALongVectorComparison( size,
                                       deviceData1,
                                       deviceData2 );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
}

#endif /* MEMORYFUNCTIONS_H_ */
