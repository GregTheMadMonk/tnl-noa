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
#include <core/cuda/cuda-reduction.h>
#include <core/cuda/reduction-operations.h>
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
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
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
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return true;
#endif
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
                                          const Element value )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index gridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      data[ elementIdx ] = value;
      elementIdx += gridSize;
   }
}
#endif

template< typename Element, typename Index >
bool setMemoryCuda( Element* data,
                    const Element& value,
                    const Index size )
{
#ifdef HAVE_CUDA
   dim3 blockSize( 0 ), gridSize( 0 );
   blockSize. x = 256;
   Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
   gridSize. x = Min( blocksNumber, ( Index ) maxCudaGridSize - 1 );
   setVectorValueCudaKernel<<< gridSize, blockSize >>>( data, size, value );

   return checkCudaDevice;
#else
      cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
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
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
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
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
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
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
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
   bool result;
   tnlParallelReductionEqualities< Element, Index > operation;
   reductionOnCudaDevice( operation,
                          size,
                          deviceData1,
                          deviceData2,
                          result );
   return result;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template bool allocateMemoryHost( char*& data, const int size );
extern template bool allocateMemoryHost( int*& data, const int size );
extern template bool allocateMemoryHost( long int*& data, const int size );
extern template bool allocateMemoryHost( float*& data, const int size );
extern template bool allocateMemoryHost( double*& data, const int size );
extern template bool allocateMemoryHost( long double*& data, const int size );

extern template bool allocateMemoryHost( char*& data, const long int size );
extern template bool allocateMemoryHost( int*& data, const long int size );
extern template bool allocateMemoryHost( long int*& data, const long int size );
extern template bool allocateMemoryHost( float*& data, const long int size );
extern template bool allocateMemoryHost( double*& data, const long int size );
extern template bool allocateMemoryHost( long double*& data, const long int size );

extern template bool allocateMemoryCuda( char*& data, const int size );
extern template bool allocateMemoryCuda( int*& data, const int size );
extern template bool allocateMemoryCuda( long int*& data, const int size );
extern template bool allocateMemoryCuda( float*& data, const int size );
extern template bool allocateMemoryCuda( double*& data, const int size );
extern template bool allocateMemoryCuda( long double*& data, const int size );

extern template bool allocateMemoryCuda( char*& data, const long int size );
extern template bool allocateMemoryCuda( int*& data, const long int size );
extern template bool allocateMemoryCuda( long int*& data, const long int size );
extern template bool allocateMemoryCuda( float*& data, const long int size );
extern template bool allocateMemoryCuda( double*& data, const long int size );
extern template bool allocateMemoryCuda( long double*& data, const long int size );

extern template bool freeMemoryHost( char* data );
extern template bool freeMemoryHost( int* data );
extern template bool freeMemoryHost( long int* data );
extern template bool freeMemoryHost( float* data );
extern template bool freeMemoryHost( double* data );
extern template bool freeMemoryHost( long double* data );

extern template bool freeMemoryCuda( char* data );
extern template bool freeMemoryCuda( int* data );
extern template bool freeMemoryCuda( long int* data );
extern template bool freeMemoryCuda( float* data );
extern template bool freeMemoryCuda( double* data );
extern template bool freeMemoryCuda( long double* data );

extern template bool setMemoryHost( char* data, const char& value, const int size );
extern template bool setMemoryHost( int* data, const int& value, const int size );
extern template bool setMemoryHost( long int* data, const long int& value, const int size );
extern template bool setMemoryHost( float* data, const float& value, const int size );
extern template bool setMemoryHost( double* data, const double& value, const int size );
extern template bool setMemoryHost( long double* data, const long double& value, const int size );

extern template bool setMemoryHost( char* data, const char& value, const long int size );
extern template bool setMemoryHost( int* data, const int& value, const long int size );
extern template bool setMemoryHost( long int* data, const long int& value, const long int size );
extern template bool setMemoryHost( float* data, const float& value, const long int size );
extern template bool setMemoryHost( double* data, const double& value, const long int size );
extern template bool setMemoryHost( long double* data, const long double& value, const long int size );

extern template bool setMemoryCuda( char* data, const char& value, const int size );
extern template bool setMemoryCuda( int* data, const int& value, const int size );
extern template bool setMemoryCuda( long int* data, const long int& value, const int size );
extern template bool setMemoryCuda( float* data, const float& value, const int size );
extern template bool setMemoryCuda( double* data, const double& value, const int size );
extern template bool setMemoryCuda( long double* data, const long double& value, const int size );

extern template bool setMemoryCuda( char* data, const char& value, const long int size );
extern template bool setMemoryCuda( int* data, const int& value, const long int size );
extern template bool setMemoryCuda( long int* data, const long int& value, const long int size );
extern template bool setMemoryCuda( float* data, const float& value, const long int size );
extern template bool setMemoryCuda( double* data, const double& value, const long int size );
extern template bool setMemoryCuda( long double* data, const long double& value, const long int size );

extern template bool copyMemoryHostToHost( char* destination, const char* source, const int size );
extern template bool copyMemoryHostToHost( int* destination, const int* source, const int size );
extern template bool copyMemoryHostToHost( long int* destination, const long int* source, const int size );
extern template bool copyMemoryHostToHost( float* destination, const float* source, const int size );
extern template bool copyMemoryHostToHost( double* destination, const double* source, const int size );
extern template bool copyMemoryHostToHost( long double* destination, const long double* source, const int size );

extern template bool copyMemoryHostToHost( char* destination, const char* source, const long int size );
extern template bool copyMemoryHostToHost( int* destination, const int* source, const long int size );
extern template bool copyMemoryHostToHost( long int* destination, const long int* source, const long int size );
extern template bool copyMemoryHostToHost( float* destination, const float* source, const long int size );
extern template bool copyMemoryHostToHost( double* destination, const double* source, const long int size );
extern template bool copyMemoryHostToHost( long double* destination, const long double* source, const long int size );

extern template bool copyMemoryCudaToHost( char* destination, const char* source, const int size );
extern template bool copyMemoryCudaToHost( int* destination, const int* source, const int size );
extern template bool copyMemoryCudaToHost( long int* destination, const long int* source, const int size );
extern template bool copyMemoryCudaToHost( float* destination, const float* source, const int size );
extern template bool copyMemoryCudaToHost( double* destination, const double* source, const int size );

extern template bool copyMemoryCudaToHost( char* destination, const char* source, const long int size );
extern template bool copyMemoryCudaToHost( int* destination, const int* source, const long int size );
extern template bool copyMemoryCudaToHost( long int* destination, const long int* source, const long int size );
extern template bool copyMemoryCudaToHost( float* destination, const float* source, const long int size );
extern template bool copyMemoryCudaToHost( double* destination, const double* source, const long int size );

extern template bool copyMemoryHostToCuda( char* destination, const char* source, const int size );
extern template bool copyMemoryHostToCuda( int* destination, const int* source, const int size );
extern template bool copyMemoryHostToCuda( long int* destination, const long int* source, const int size );
extern template bool copyMemoryHostToCuda( float* destination, const float* source, const int size );
extern template bool copyMemoryHostToCuda( double* destination, const double* source, const int size );

extern template bool copyMemoryHostToCuda( char* destination, const char* source, const long int size );
extern template bool copyMemoryHostToCuda( int* destination, const int* source, const long int size );
extern template bool copyMemoryHostToCuda( long int* destination, const long int* source, const long int size );
extern template bool copyMemoryHostToCuda( float* destination, const float* source, const long int size );
extern template bool copyMemoryHostToCuda( double* destination, const double* source, const long int size );

extern template bool copyMemoryCudaToCuda( char* destination, const char* source, const int size );
extern template bool copyMemoryCudaToCuda( int* destination, const int* source, const int size );
extern template bool copyMemoryCudaToCuda( long int* destination, const long int* source, const int size );
extern template bool copyMemoryCudaToCuda( float* destination, const float* source, const int size );
extern template bool copyMemoryCudaToCuda( double* destination, const double* source, const int size );

extern template bool copyMemoryCudaToCuda( char* destination, const char* source, const long int size );
extern template bool copyMemoryCudaToCuda( int* destination, const int* source, const long int size );
extern template bool copyMemoryCudaToCuda( long int* destination, const long int* source, const long int size );
extern template bool copyMemoryCudaToCuda( float* destination, const float* source, const long int size );
extern template bool copyMemoryCudaToCuda( double* destination, const double* source, const long int size );

extern template bool compareMemoryHost( const char* data1, const char* data2, const int size );
extern template bool compareMemoryHost( const int* data1, const int* data2, const int size );
extern template bool compareMemoryHost( const long int* data1, const long int* data2, const int size );
extern template bool compareMemoryHost( const float* data1, const float* data2, const int size );
extern template bool compareMemoryHost( const double* data1, const double* data2, const int size );
extern template bool compareMemoryHost( const long double* data1, const long double* data2, const int size );

extern template bool compareMemoryHost( const char* data1, const char* data2, const long int size );
extern template bool compareMemoryHost( const int* data1, const int* data2, const long int size );
extern template bool compareMemoryHost( const long int* data1, const long int* data2, const long int size );
extern template bool compareMemoryHost( const float* data1, const float* data2, const long int size );
extern template bool compareMemoryHost( const double* data1, const double* data2, const long int size );
extern template bool compareMemoryHost( const long double* data1, const long double* data2, const long int size );

extern template bool compareMemoryHostCuda( const char* data1, const char* data2, const int size );
extern template bool compareMemoryHostCuda( const int* data1, const int* data2, const int size );
extern template bool compareMemoryHostCuda( const long int* data1, const long int* data2, const int size );
extern template bool compareMemoryHostCuda( const float* data1, const float* data2, const int size );
extern template bool compareMemoryHostCuda( const double* data1, const double* data2, const int size );
extern template bool compareMemoryHostCuda( const long double* data1, const long double* data2, const int size );

extern template bool compareMemoryHostCuda( const char* data1, const char* data2, const long int size );
extern template bool compareMemoryHostCuda( const int* data1, const int* data2, const long int size );
extern template bool compareMemoryHostCuda( const long int* data1, const long int* data2, const long int size );
extern template bool compareMemoryHostCuda( const float* data1, const float* data2, const long int size );
extern template bool compareMemoryHostCuda( const double* data1, const double* data2, const long int size );
extern template bool compareMemoryHostCuda( const long double* data1, const long double* data2, const long int size );

extern template bool compareMemoryCuda( const char* data1, const char* data2, const int size );
extern template bool compareMemoryCuda( const int* data1, const int* data2, const int size );
extern template bool compareMemoryCuda( const long int* data1, const long int* data2, const int size );
extern template bool compareMemoryCuda( const float* data1, const float* data2, const int size );
extern template bool compareMemoryCuda( const double* data1, const double* data2, const int size );
extern template bool compareMemoryCuda( const long double* data1, const long double* data2, const int size );

extern template bool compareMemoryCuda( const char* data1, const char* data2, const long int size );
extern template bool compareMemoryCuda( const int* data1, const int* data2, const long int size );
extern template bool compareMemoryCuda( const long int* data1, const long int* data2, const long int size );
extern template bool compareMemoryCuda( const float* data1, const float* data2, const long int size );
extern template bool compareMemoryCuda( const double* data1, const double* data2, const long int size );
extern template bool compareMemoryCuda( const long double* data1, const long double* data2, const long int size );

#endif

#endif /* MEMORYFUNCTIONS_H_ */
