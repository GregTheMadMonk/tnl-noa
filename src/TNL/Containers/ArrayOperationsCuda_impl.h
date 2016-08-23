/***************************************************************************
                          ArrayOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <iostream>
#include <TNL/tnlConfig.h>
#include <TNL/core/mfuncs.h>
#include <TNL/Containers/Algorithms/cuda-reduction.h>
#include <TNL/Containers/Algorithms/reduction-operations.h>

namespace TNL {
namespace Containers {   

template< typename Element, typename Index >
bool ArrayOperations< Devices::Cuda >::allocateMemory( Element*& data,
                                                    const Index size )
{
#ifdef HAVE_CUDA
   if( cudaMalloc( ( void** ) &data,
                   ( size_t ) size * sizeof( Element ) ) != cudaSuccess )
      data = 0;
   return checkCudaDevice;
#else
   CudaSupportMissingMessage;
   return false;
#endif
}

template< typename Element >
bool ArrayOperations< Devices::Cuda >::freeMemory( Element* data )
{
   Assert( data, );
#ifdef HAVE_CUDA
      cudaFree( data );
      return checkCudaDevice;
#else
      CudaSupportMissingMessage;;
   return true;
#endif
}

template< typename Element >
void ArrayOperations< Devices::Cuda >::setMemoryElement( Element* data,
                                                      const Element& value )
{
   Assert( data, );
   ArrayOperations< Devices::Cuda >::setMemory( data, value, 1 );
}

template< typename Element >
Element ArrayOperations< Devices::Cuda >::getMemoryElement( const Element* data )
{
   Assert( data, );
   Element result;
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< Element, Element, int >( &result, data, 1 );
   return result;
}

template< typename Element, typename Index >
Element& ArrayOperations< Devices::Cuda >::getArrayElementReference( Element* data, const Index i )
{
   Assert( data, );
   return data[ i ];
}

template< typename Element, typename Index >
const Element& ArrayOperations< Devices::Cuda >::getArrayElementReference( const Element* data, const Index i )
{
   Assert( data, );
   return data[ i ];
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
bool ArrayOperations< Devices::Cuda >::setMemory( Element* data,
                    const Element& value,
                    const Index size )
{
   Assert( data, );
#ifdef HAVE_CUDA
   dim3 blockSize( 0 ), gridSize( 0 );
   blockSize. x = 256;
   Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
   gridSize. x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
   setArrayValueCudaKernel<<< gridSize, blockSize >>>( data, size, value );
   return checkCudaDevice;
#else
   CudaSupportMissingMessage;;
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
bool ArrayOperations< Devices::Cuda >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   Assert( destination, );
   Assert( source, );
   #ifdef HAVE_CUDA
      if( std::is_same< DestinationElement, SourceElement >::value )
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
         gridSize. x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
         copyMemoryCudaToCudaKernel<<< gridSize, blockSize >>>( destination, source, size );
         return checkCudaDevice;
      }
   #else
      CudaSupportMissingMessage;;
   #endif
      return false;
}

template< typename Element1,
          typename Element2,
          typename Index >
bool ArrayOperations< Devices::Cuda >::compareMemory( const Element1* destination,
                                                   const Element2* source,
                                                   const Index size )
{
   Assert( destination, );
   Assert( source, );
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
bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   Assert( destination, );
   Assert( source, );
   #ifdef HAVE_CUDA
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyDeviceToHost );
      if( ! checkCudaDevice )
      {
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
         return false;
      }
      return true;
   }
   else
   {
      SourceElement* buffer = new SourceElement[ Devices::Cuda::getGPUTransferBufferSize() ];
      if( ! buffer )
      {
         std::cerr << "Unable to allocate supporting buffer to transfer data between the CUDA device and the host." << std::endl;
         return false;
      }
      Index i( 0 );
      while( i < size )
      {
         if( cudaMemcpy( buffer,
                         &source[ i ],
                         min( size - i, Devices::Cuda::getGPUTransferBufferSize() ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost ) != cudaSuccess )
         {
            checkCudaDevice;
            delete[] buffer;
            return false;
         }
         Index j( 0 );
         while( j < Devices::Cuda::getGPUTransferBufferSize() && i + j < size )
            destination[ i + j ] = buffer[ j++ ];
         i += j;
      }
      delete[] buffer;
   }
   #else
      CudaSupportMissingMessage;;
      return false;
   #endif
   return true;
}


template< typename Element1,
          typename Element2,
          typename Index >
bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory( const Element1* destination,
                                                            const Element2* source,
                                                            const Index size )
{
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   Assert( destination, );
   Assert( source, );
   Assert( size >= 0, std::cerr << "size = " << size );
   #ifdef HAVE_CUDA
   Element2* host_buffer = new Element2[ Devices::Cuda::getGPUTransferBufferSize() ];
   if( ! host_buffer )
   {
      std::cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << std::endl;
      return false;
   }
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = min( size - compared, Devices::Cuda::getGPUTransferBufferSize() );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( source[ compared ] ),
                      transfer * sizeof( Element2 ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         std::cerr << "Transfer of data from the device failed." << std::endl;
         checkCudaDevice;
         delete[] host_buffer;
         return false;
      }
      if( ! ArrayOperations< Devices::Host >::compareMemory( &destination[ compared ], host_buffer, transfer ) )
      {
         delete[] host_buffer;
         return false;
      }
      compared += transfer;
   }
   delete[] host_buffer;
   return true;
   #else
      CudaSupportMissingMessage;;
      return false;
   #endif
}

/****
 * Operations Host -> CUDA
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( DestinationElement* destination,
                                                         const SourceElement* source,
                                                         const Index size )
{
   Assert( destination, );
   Assert( source, );
   Assert( size >= 0, std::cerr << "size = " << size );
   #ifdef HAVE_CUDA
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      cudaMemcpy( destination,
                  source,
                  size * sizeof( DestinationElement ),
                  cudaMemcpyHostToDevice );
      if( ! checkCudaDevice )
      {
         std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
         return false;
      }
      return true;
   }
   else
   {
      DestinationElement* buffer = new DestinationElement[ Devices::Cuda::getGPUTransferBufferSize() ];
      if( ! buffer )
      {
         std::cerr << "Unable to allocate supporting buffer to transfer data between the CUDA device and the host." << std::endl;
         return false;
      }
      Index i( 0 );
      while( i < size )
      {
         Index j( 0 );
         while( j < Devices::Cuda::getGPUTransferBufferSize() && i + j < size )
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
      CudaSupportMissingMessage;;
      return false;
   #endif
}

template< typename Element1,
          typename Element2,
          typename Index >
bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory( const Element1* hostData,
                                                            const Element2* deviceData,
                                                            const Index size )
{
   Assert( hostData, );
   Assert( deviceData, );
   Assert( size >= 0, std::cerr << "size = " << size );
   return ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory( deviceData, hostData, size );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< char,        int >( char*& data, const int size );
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< int,         int >( int*& data, const int size );
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< long int,    int >( long int*& data, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< float,       int >( float*& data, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< double,      int >( double*& data, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< long double, int >( long double*& data, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< char,        long int >( char*& data, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< int,         long int >( int*& data, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< long int,    long int >( long int*& data, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< float,       long int >( float*& data, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< double,      long int >( double*& data, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::allocateMemory< long double, long int >( long double*& data, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda >::freeMemory< char        >( char* data );
extern template bool ArrayOperations< Devices::Cuda >::freeMemory< int         >( int* data );
extern template bool ArrayOperations< Devices::Cuda >::freeMemory< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::freeMemory< float       >( float* data );
#endif
extern template bool ArrayOperations< Devices::Cuda >::freeMemory< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::freeMemory< long double >( long double* data );
#endif

extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< char        >( char* data, const char& value );
extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< int         >( int* data, const int& value );
extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< long int    >( long int* data, const long int& value );
#ifdef INSTANTIATE_FLOAT
extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< float       >( float* data, const float& value );
#endif
extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< double      >( double* data, const double& value );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template void ArrayOperations< Devices::Cuda >::setMemoryElement< long double >( long double* data, const long double& value );
#endif

extern template char        ArrayOperations< Devices::Cuda >::getMemoryElement< char        >( const char* data );
extern template int         ArrayOperations< Devices::Cuda >::getMemoryElement< int         >( const int* data );
extern template long int    ArrayOperations< Devices::Cuda >::getMemoryElement< long int    >( const long int* data );
#ifdef INSTANTIATE_FLOAT
extern template float       ArrayOperations< Devices::Cuda >::getMemoryElement< float       >( const float* data );
#endif
extern template double      ArrayOperations< Devices::Cuda >::getMemoryElement< double      >( const double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double ArrayOperations< Devices::Cuda >::getMemoryElement< long double >( const long double* data );
#endif

extern template char&        ArrayOperations< Devices::Cuda >::getArrayElementReference< char,        int >( char* data, const int i );
extern template int&         ArrayOperations< Devices::Cuda >::getArrayElementReference< int,         int >( int* data, const int i );
extern template long int&    ArrayOperations< Devices::Cuda >::getArrayElementReference< long int,    int >( long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
extern template float&       ArrayOperations< Devices::Cuda >::getArrayElementReference< float,       int >( float* data, const int i );
#endif
extern template double&      ArrayOperations< Devices::Cuda >::getArrayElementReference< double,      int >( double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double& ArrayOperations< Devices::Cuda >::getArrayElementReference< long double, int >( long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template char&        ArrayOperations< Devices::Cuda >::getArrayElementReference< char,        long int >( char* data, const long int i );
extern template int&         ArrayOperations< Devices::Cuda >::getArrayElementReference< int,         long int >( int* data, const long int i );
extern template long int&    ArrayOperations< Devices::Cuda >::getArrayElementReference< long int,    long int >( long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
extern template float&       ArrayOperations< Devices::Cuda >::getArrayElementReference< float,       long int >( float* data, const long int i );
#endif
extern template double&      ArrayOperations< Devices::Cuda >::getArrayElementReference< double,      long int >( double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double& ArrayOperations< Devices::Cuda >::getArrayElementReference< long double, long int >( long double* data, const long int i );
#endif
#endif

extern template const char&        ArrayOperations< Devices::Cuda >::getArrayElementReference< char,        int >( const char* data, const int i );
extern template const int&         ArrayOperations< Devices::Cuda >::getArrayElementReference< int,         int >( const int* data, const int i );
extern template const long int&    ArrayOperations< Devices::Cuda >::getArrayElementReference< long int,    int >( const long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
extern template const float&       ArrayOperations< Devices::Cuda >::getArrayElementReference< float,       int >( const float* data, const int i );
#endif
extern template const double&      ArrayOperations< Devices::Cuda >::getArrayElementReference< double,      int >( const double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template const long double& ArrayOperations< Devices::Cuda >::getArrayElementReference< long double, int >( const long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template const char&        ArrayOperations< Devices::Cuda >::getArrayElementReference< char,        long int >( const char* data, const long int i );
extern template const int&         ArrayOperations< Devices::Cuda >::getArrayElementReference< int,         long int >( const int* data, const long int i );
extern template const long int&    ArrayOperations< Devices::Cuda >::getArrayElementReference< long int,    long int >( const long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
extern template const float&       ArrayOperations< Devices::Cuda >::getArrayElementReference< float,       long int >( const float* data, const long int i );
#endif
extern template const double&      ArrayOperations< Devices::Cuda >::getArrayElementReference< double,      long int >( const double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template const long double& ArrayOperations< Devices::Cuda >::getArrayElementReference< long double, long int >( const long double* data, const long int i );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

extern template bool ArrayOperations< Devices::Cuda >::setMemory< char,        int >( char* destination, const char& value, const int size );
extern template bool ArrayOperations< Devices::Cuda >::setMemory< int,         int >( int* destination, const int& value, const int size );
extern template bool ArrayOperations< Devices::Cuda >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::setMemory< float,       int >( float* destination, const float& value, const int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::setMemory< double,      int >( double* destination, const double& value, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool ArrayOperations< Devices::Cuda >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
extern template bool ArrayOperations< Devices::Cuda >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool ArrayOperations< Devices::Cuda >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
#endif
extern template bool ArrayOperations< Devices::Cuda >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool ArrayOperations< Devices::Cuda >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
