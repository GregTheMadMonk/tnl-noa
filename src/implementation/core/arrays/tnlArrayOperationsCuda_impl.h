/***************************************************************************
                          tnlArrayOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLARRAYOPERATIONSCUDA_IMPL_H_
#define TNLARRAYOPERATIONSCUDA_IMPL_H_

#include <iostream>
#include <tnlConfig.h>
#include <core/mfuncs.h>
#include <core/cuda/cuda-reduction.h>
#include <core/cuda/reduction-operations.h>


template< typename Element, typename Index >
bool tnlArrayOperations< tnlCuda >::allocateMemory( Element*& data,
                     const Index size )
{
#ifdef HAVE_CUDA
   if( cudaMalloc( ( void** ) &data,
                   ( size_t ) size * sizeof( Element ) ) != cudaSuccess )
      data = 0;
   return checkCudaDevice;
#else
   tnlCudaSupportMissingMessage;
   return false;
#endif
}

template< typename Element >
bool tnlArrayOperations< tnlCuda >::freeMemory( Element* data )
{
#ifdef HAVE_CUDA
      cudaFree( data );
      return checkCudaDevice;
#else
      tnlCudaSupportMissingMessage;;
   return true;
#endif
}

template< typename Element >
void tnlArrayOperations< tnlCuda >::setMemoryElement( Element* data,
                                                        const Element& value )
{
   tnlArrayOperations< tnlCuda >::setMemory( data, value, 1 );
}

template< typename Element >
Element tnlArrayOperations< tnlCuda >::getMemoryElement( const Element* data )
{
   Element result;
   tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< Element, Element, int >( &result, data, 1 );
   return result;
}

template< typename Element, typename Index >
Element& tnlArrayOperations< tnlCuda >::getArrayElementReference( Element* data, const Index i )
{
   // TODO: implement this
   tnlAssert( false, cerr << "Implement this" << endl );
}

template< typename Element, typename Index >
const Element& tnlArrayOperations< tnlCuda >::getArrayElementReference(const Element* data, const Index i )
{
   // TODO: implement this
   tnlAssert( false, cerr << "Implement this" << endl );
}


#ifdef HAVE_CUDA
template< typename Element, typename Index >
__global__ void setArrayValueCudaKernel( Element* data,
                                         const Index size,
                                         const Element value )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      data[ elementIdx ] = value;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Element, typename Index >
bool tnlArrayOperations< tnlCuda >::setMemory( Element* data,
                    const Element& value,
                    const Index size )
{
#ifdef HAVE_CUDA
   dim3 blockSize( 0 ), gridSize( 0 );
   blockSize. x = 256;
   Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
   gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
   setArrayValueCudaKernel<<< gridSize, blockSize >>>( data, size, value );
   return checkCudaDevice;
#else
      tnlCudaSupportMissingMessage;;
      return false;
#endif
}

#ifdef HAVE_CUDA
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
__global__ void copyMemoryCudaToCudaKernel( DestinationElement* destination,
                                            const SourceElement* source,
                                            const Index size )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      destination[ elementIdx ] = source[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlCuda >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   #ifdef HAVE_CUDA
      if( tnlFastArrayOperations< DestinationElement, SourceElement >::enabled )
      {
         if( cudaMemcpy( destination,
                         source,
                         size * sizeof( DestinationElement ),
                         cudaMemcpyDeviceToDevice ) != cudaSuccess )
         return checkCudaDevice;
      }
      else
      {
         dim3 blockSize( 0 ), gridSize( 0 );
         blockSize. x = 256;
         Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
         gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
         copyMemoryCudaToCudaKernel<<< gridSize, blockSize >>>( destination, source, size );
         return checkCudaDevice;
      }
   #else
         tnlCudaSupportMissingMessage;;
         return false;
   #endif
}

template< typename Element1,
          typename Element2,
          typename Index >
bool tnlArrayOperations< tnlCuda >::compareMemory( const Element1* destination,
                                                   const Element2* source,
                                                   const Index size )
{
   //TODO: The parallel reduction on the CUDA device with different element types is needed.
   bool result;
   tnlParallelReductionEqualities< Element1, Index > reductionEqualities;
   reductionOnCudaDevice( reductionEqualities, size, destination, source, result );
   return result;
}

/****
 * Operations CUDA -> Host
 */

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   #ifdef HAVE_CUDA
   if( tnlFastArrayOperations< DestinationElement, SourceElement >::enabled )
   {
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyDeviceToHost );
      if( ! checkCudaDevice )
      {
         cerr << "Transfer of data from CUDA device to host failed." << endl;
         return false;
      }
      return true;
   }
   else
   {
      SourceElement* buffer = new SourceElement[ tnlCuda::getGPUTransferBufferSize() ];
      if( ! buffer )
      {
         cerr << "Unable to allocate supporting buffer to transfer data between the CUDA device and the host." << endl;
         return false;
      }
      Index i( 0 );
      while( i < size )
      {
         if( cudaMemcpy( buffer,
                         &source[ i ],
                         Min( size - i, tnlCuda::getGPUTransferBufferSize() ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost ) != cudaSuccess )
         {
            checkCudaDevice;
            delete[] buffer;
            return false;
         }
         Index j( 0 );
         while( j < tnlCuda::getGPUTransferBufferSize() && i + j < size )
            destination[ i + j ] = buffer[ j++ ];
         i += j;
      }
      delete[] buffer;
   }
   #else
      tnlCudaSupportMissingMessage;;
      return false;
   #endif
}


template< typename Element1,
          typename Element2,
          typename Index >
bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory( const Element1* destination,
                                                            const Element2* source,
                                                            const Index size )
{
   #ifdef HAVE_CUDA
   Element2* host_buffer = new Element2[ tnlCuda::getGPUTransferBufferSize() ];
   if( ! host_buffer )
   {
      cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << endl;
      return false;
   }
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = Min( size - compared, tnlCuda::getGPUTransferBufferSize() );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( source[ compared ] ),
                      transfer * sizeof( Element2 ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         cerr << "Transfer of data from the device failed." << endl;
         checkCudaDevice;
         delete[] host_buffer;
         return false;
      }
      if( ! tnlArrayOperations< tnlHost >::compareMemory( host_buffer, destination, transfer ) )
      {
         delete[] host_buffer;
         return false;
      }
      compared += transfer;
   }
   delete[] host_buffer;
   return true;
   #else
      tnlCudaSupportMissingMessage;;
      return false;
   #endif
}

/****
 * Operations Host -> CUDA
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   #ifdef HAVE_CUDA
   if( tnlFastArrayOperations< DestinationElement, SourceElement >::enabled )
   {
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyHostToDevice );
      if( ! checkCudaDevice )
      {
         cerr << "Transfer of data from host to CUDA device failed." << endl;
         return false;
      }
      return true;
   }
   else
   {
      DestinationElement* buffer = new DestinationElement[ tnlCuda::getGPUTransferBufferSize() ];
      if( ! buffer )
      {
         cerr << "Unable to allocate supporting buffer to transfer data between the CUDA device and the host." << endl;
         return false;
      }
      Index i( 0 );
      while( i < size )
      {
         Index j( 0 );
         while( j < tnlCuda::getGPUTransferBufferSize() && i + j < size )
            buffer[ j ] = source[ i + j++ ];
         if( cudaMemcpy( &destination[ i ],
                         buffer,
                         j * sizeof( DestinationElement ),
                         cudaMemcpyHostToDevice ) != cudaSuccess )
         {
            checkCudaDevice;
            delete[] buffer;
            return false;
         }
         i += j;
      }
      delete[] buffer;
      return true;
   }
   #else
      tnlCudaSupportMissingMessage;;
      return false;
   #endif
}

template< typename Element1,
          typename Element2,
          typename Index >
bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory( const Element1* hostData,
                                                            const Element2* deviceData,
                                                            const Index size )
{
   return tnlArrayOperations< tnlHost, tnlCuda >::compareMemory( deviceData, hostData, size );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< char,        int >( char*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< int,         int >( int*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< long int,    int >( long int*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< float,       int >( float*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< double,      int >( double*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< long double, int >( long double*& data, const int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< char,        long int >( char*& data, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< int,         long int >( int*& data, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< long int,    long int >( long int*& data, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< float,       long int >( float*& data, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< double,      long int >( double*& data, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::allocateMemory< long double, long int >( long double*& data, const long int size );

extern template bool tnlArrayOperations< tnlCuda >::freeMemory< char        >( char* data );
extern template bool tnlArrayOperations< tnlCuda >::freeMemory< int         >( int* data );
extern template bool tnlArrayOperations< tnlCuda >::freeMemory< long int    >( long int* data );
extern template bool tnlArrayOperations< tnlCuda >::freeMemory< float       >( float* data );
extern template bool tnlArrayOperations< tnlCuda >::freeMemory< double      >( double* data );
extern template bool tnlArrayOperations< tnlCuda >::freeMemory< long double >( long double* data );

extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< char        >( char* data, const char& value );
extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< int         >( int* data, const int& value );
extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< long int    >( long int* data, const long int& value );
extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< float       >( float* data, const float& value );
extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< double      >( double* data, const double& value );
extern template void tnlArrayOperations< tnlCuda >::setMemoryElement< long double >( long double* data, const long double& value );

extern template char        tnlArrayOperations< tnlCuda >::getMemoryElement< char        >( const char* data );
extern template int         tnlArrayOperations< tnlCuda >::getMemoryElement< int         >( const int* data );
extern template long int    tnlArrayOperations< tnlCuda >::getMemoryElement< long int    >( const long int* data );
extern template float       tnlArrayOperations< tnlCuda >::getMemoryElement< float       >( const float* data );
extern template double      tnlArrayOperations< tnlCuda >::getMemoryElement< double      >( const double* data );
extern template long double tnlArrayOperations< tnlCuda >::getMemoryElement< long double >( const long double* data );

extern template char&        tnlArrayOperations< tnlCuda >::getArrayElementReference< char,        int >( char* data, const int i );
extern template int&         tnlArrayOperations< tnlCuda >::getArrayElementReference< int,         int >( int* data, const int i );
extern template long int&    tnlArrayOperations< tnlCuda >::getArrayElementReference< long int,    int >( long int* data, const int i );
extern template float&       tnlArrayOperations< tnlCuda >::getArrayElementReference< float,       int >( float* data, const int i );
extern template double&      tnlArrayOperations< tnlCuda >::getArrayElementReference< double,      int >( double* data, const int i );
extern template long double& tnlArrayOperations< tnlCuda >::getArrayElementReference< long double, int >( long double* data, const int i );

extern template char&        tnlArrayOperations< tnlCuda >::getArrayElementReference< char,        long int >( char* data, const long int i );
extern template int&         tnlArrayOperations< tnlCuda >::getArrayElementReference< int,         long int >( int* data, const long int i );
extern template long int&    tnlArrayOperations< tnlCuda >::getArrayElementReference< long int,    long int >( long int* data, const long int i );
extern template float&       tnlArrayOperations< tnlCuda >::getArrayElementReference< float,       long int >( float* data, const long int i );
extern template double&      tnlArrayOperations< tnlCuda >::getArrayElementReference< double,      long int >( double* data, const long int i );
extern template long double& tnlArrayOperations< tnlCuda >::getArrayElementReference< long double, long int >( long double* data, const long int i );

extern template const char&        tnlArrayOperations< tnlCuda >::getArrayElementReference< char,        int >( const char* data, const int i );
extern template const int&         tnlArrayOperations< tnlCuda >::getArrayElementReference< int,         int >( const int* data, const int i );
extern template const long int&    tnlArrayOperations< tnlCuda >::getArrayElementReference< long int,    int >( const long int* data, const int i );
extern template const float&       tnlArrayOperations< tnlCuda >::getArrayElementReference< float,       int >( const float* data, const int i );
extern template const double&      tnlArrayOperations< tnlCuda >::getArrayElementReference< double,      int >( const double* data, const int i );
extern template const long double& tnlArrayOperations< tnlCuda >::getArrayElementReference< long double, int >( const long double* data, const int i );

extern template const char&        tnlArrayOperations< tnlCuda >::getArrayElementReference< char,        long int >( const char* data, const long int i );
extern template const int&         tnlArrayOperations< tnlCuda >::getArrayElementReference< int,         long int >( const int* data, const long int i );
extern template const long int&    tnlArrayOperations< tnlCuda >::getArrayElementReference< long int,    long int >( const long int* data, const long int i );
extern template const float&       tnlArrayOperations< tnlCuda >::getArrayElementReference< float,       long int >( const float* data, const long int i );
extern template const double&      tnlArrayOperations< tnlCuda >::getArrayElementReference< double,      long int >( const double* data, const long int i );
extern template const long double& tnlArrayOperations< tnlCuda >::getArrayElementReference< long double, long int >( const long double* data, const long int i );

extern template bool tnlArrayOperations< tnlCuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );

extern template bool tnlArrayOperations< tnlCuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
extern template bool tnlArrayOperations< tnlCuda, tnlHost >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );

extern template bool tnlArrayOperations< tnlCuda >::setMemory< char,        int >( char* destination, const char& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< int,         int >( int* destination, const int& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< float,       int >( float* destination, const float& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< double,      int >( double* destination, const double& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
extern template bool tnlArrayOperations< tnlCuda >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );

#endif

#endif /* TNLARRAYOPERATIONSCUDA_IMPL_H_ */
