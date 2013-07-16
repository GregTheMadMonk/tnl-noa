/***************************************************************************
                          tnlArrayOperationsHost_impl.h  -  description
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

#ifndef TNLARRAYOPERATIONSHOST_IMPL_H_
#define TNLARRAYOPERATIONSHOST_IMPL_H_

template< typename Element, typename Index >
bool tnlArrayOperations< tnlHost > :: allocateMemory( Element*& data,
                                                      const Index size )
{
   if( ! ( data = new Element[ size ] ) )
      return false;
   return true;
}

template< typename Element >
bool tnlArrayOperations< tnlHost > :: freeMemory( Element* data )
{
   delete[] data;
   return true;
}
template< typename Element >
void tnlArrayOperations< tnlHost > :: setMemoryElement( Element* data,
                                                        const Element& value )
{
   *data = value;
};

template< typename Element >
Element tnlArrayOperations< tnlHost > :: getMemoryElement( Element* data )
{
   return *data;
};

template< typename Element, typename Index >
Element& tnlArrayOperations< tnlHost > :: getArrayElementReference( Element* data,
                                                                    const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index >
const Element& tnlArrayOperations< tnlHost > :: getArrayElementReference( const Element* data,
                                                                          const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index >
bool tnlArrayOperations< tnlHost > :: setMemory( Element* data,
                                                 const Element& value,
                                                 const Index size )
{
   for( Index i = 0; i < size; i ++ )
      data[ i ] = value;
   return true;
}

template< typename DestinationElement,
          typename DestinationDevice,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlHost > :: copyMemory( DestinationElement* destination,
                                                  const SourceElement* source,
                                                  const Index size )
{
   if( DestinationDevice :: getDevice() == tnlHostDevice )
      for( Index i = 0; i < size; i ++ )
         destination[ i ] = ( DestinationElement) source[ i ];
   if( DestinationDevice :: getDevice() == tnlCudaDevice )
   {
      DestinationElement* buffer = new DestinationElement[ tnlGPUvsCPUTransferBufferSize ];
      if( ! buffer )
      {
         cerr << "Unable to allocate supporting buffer to transfer data between the CUDA device and the host." << endl;
         return false;
      }
      Index i( 0 );
      while( i < size )
      {
         Index j( 0 );
         while( j < tnlGPUvsCPUTransferBufferSize && i + j < size )
            buffer[ j ] = source[ i + j++ ];
         if( ! copyMemoryHostToCuda( &destination[ i ],
                                     buffer,
                                     j ) )
         {
            delete[] buffer;
            return false;
         }
         i += j;
      }
      delete[] buffer;
      return true;
   }
   return true;
}

/*template< typename Element,
          typename DestinationDevice,
          typename Index >
bool tnlArrayOperations< tnlHost > :: copyMemory( Element* destination,
                                                  const Element* source,
                                                  const Index size )
{
   if( DestinationDevice :: getDevice() == tnlHostDevice )
      memcpy( destination, source, size * sizeof( Element ) );
   if( DestinationDevice :: getDevice() == tnlCudaDevice )
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
   return true;
}*/

/*template< typename Element,
          typename DestinationDevice,
          typename Index >
bool tnlArrayOperations< tnlHost > :: compareMemory( const Element* destination,
                                                     const Element* source,
                                                     const Index size )
{
   if( DestinationDevice :: getDevice() == tnlHostDevice )
      if( memcmp( destination, source, size * sizeof( Element ) ) != 0 )
         return false;
   return true;
}*/

template< typename Element1,
          typename DestinationDevice,
          typename Element2,
          typename Index >
bool tnlArrayOperations< tnlHost > :: compareMemory( const Element1* destination,
                                                     const Element2* source,
                                                     const Index size )
{
   if( DestinationDevice :: getDevice() == tnlHostDevice )
      for( Index i = 0; i < size; i ++ )
         if( destination[ i ] != source[ i ] )
            return false;
   return true;
}

#ifdef UNDEF
template< typename Element1,
          typename Element2,
          typename Index >
bool tnlArrayOperations< tnlHost > :: compareMemoryOnCuda( const Element1* hostData,
                                                           const Element2* deviceData,
                                                           const Index size )
{
#ifdef HAVE_CUDA
   Element2* host_buffer = new Element2[ tnlGPUvsCPUTransferBufferSize ];
   if( ! host_buffer )
   {
      cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << endl;
      return false;
   }
   Index compared( 0 );
   while( compared < size )
   {
      Index transfer = Min( size - compared, tnlGPUvsCPUTransferBufferSize );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( deviceData[ compared ] ),
                      transfer * sizeof( Element2 ),
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
#endif


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template bool tnlArrayOperations< tnlHost >::allocateMemory< char,        int >( char*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< int,         int >( int*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long int,    int >( long int*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< float,       int >( float*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< double,      int >( double*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long double, int >( long double*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< char,        long int >( char*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< int,         long int >( int*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long int,    long int >( long int*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< float,       long int >( float*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< double,      long int >( double*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long double, long int >( long double*& data, const long int size );

extern template bool tnlArrayOperations< tnlHost >::freeMemory< char        >( char* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< int         >( int* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< long int    >( long int* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< float       >( float* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< double      >( double* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< long double >( long double* data );

extern template void tnlArrayOperations< tnlHost >::setMemoryElement< char        >( char* data, const char& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< int         >( int* data, const int& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< long int    >( long int* data, const long int& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< float       >( float* data, const float& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< double      >( double* data, const double& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< long double >( long double* data, const long double& value );

extern template char        tnlArrayOperations< tnlHost >::getMemoryElement< char        >( char* data );
extern template int         tnlArrayOperations< tnlHost >::getMemoryElement< int         >( int* data );
extern template long int    tnlArrayOperations< tnlHost >::getMemoryElement< long int    >( long int* data );
extern template float       tnlArrayOperations< tnlHost >::getMemoryElement< float       >( float* data );
extern template double      tnlArrayOperations< tnlHost >::getMemoryElement< double      >( double* data );
extern template long double tnlArrayOperations< tnlHost >::getMemoryElement< long double >( long double* data );

extern template char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        int >( char* data, const int i );
extern template int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         int >( int* data, const int i );
extern template long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    int >( long int* data, const int i );
extern template float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       int >( float* data, const int i );
extern template double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      int >( double* data, const int i );
extern template long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, int >( long double* data, const int i );

extern template char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        long int >( char* data, const long int i );
extern template int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         long int >( int* data, const long int i );
extern template long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    long int >( long int* data, const long int i );
extern template float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       long int >( float* data, const long int i );
extern template double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      long int >( double* data, const long int i );
extern template long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, long int >( long double* data, const long int i );

extern template const char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        int >( const char* data, const int i );
extern template const int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         int >( const int* data, const int i );
extern template const long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    int >( const long int* data, const int i );
extern template const float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       int >( const float* data, const int i );
extern template const double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      int >( const double* data, const int i );
extern template const long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, int >( const long double* data, const int i );

extern template const char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        long int >( const char* data, const long int i );
extern template const int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         long int >( const int* data, const long int i );
extern template const long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    long int >( const long int* data, const long int i );
extern template const float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       long int >( const float* data, const long int i );
extern template const double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      long int >( const double* data, const long int i );
extern template const long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, long int >( const long double* data, const long int i );

extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,        tnlHost,        char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,         tnlHost,         int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,    tnlHost,    long int, int >( long int* destination, const long int* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,       tnlHost,       float, int >( float* destination, const float* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,      tnlHost,      double, int >( double* destination, const double* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double, tnlHost, long double, int >( long double* destination, const long double* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,        tnlHost,        char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,         tnlHost,         int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,    tnlHost,    long int, long int >( long int* destination, const long int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,       tnlHost,       float, long int >( float* destination, const float* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,      tnlHost,      double, long int >( double* destination, const double* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double, tnlHost, long double, long int >( long double* destination, const long double* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,        tnlCuda,        char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,         tnlCuda,         int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,    tnlCuda,    long int, int >( long int* destination, const long int* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,       tnlCuda,       float, int >( float* destination, const float* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,      tnlCuda,      double, int >( double* destination, const double* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double, tnlCuda, long double, int >( long double* destination, const long double* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,        tnlCuda,        char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,         tnlCuda,         int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,    tnlCuda,    long int, long int >( long int* destination, const long int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,       tnlCuda,       float, long int >( float* destination, const float* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,      tnlCuda,      double, long int >( double* destination, const double* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double, tnlCuda, long double, long int >( long double* destination, const long double* source, const long int size );

extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,        tnlHost,        char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,         tnlHost,         int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,    tnlHost,    long int, int >( const long int* data1, const long int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,       tnlHost,       float, int >( const float* data1, const float* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,      tnlHost,      double, int >( const double* data1, const double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, tnlHost, long double, int >( const long double* data1, const long double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,        tnlHost,        char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,         tnlHost,         int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,    tnlHost,    long int, long int >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,       tnlHost,       float, long int >( const float* data1, const float* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,      tnlHost,      double, long int >( const double* data1, const double* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, tnlHost, long double, long int >( const long double* data1, const long double* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,        tnlCuda,        char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,         tnlCuda,         int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,    tnlCuda,    long int, int >( const long int* data1, const long int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,       tnlCuda,       float, int >( const float* data1, const float* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,      tnlCuda,      double, int >( const double* data1, const double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, tnlCuda, long double, int >( const long double* data1, const long double* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,        tnlCuda,        char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,         tnlCuda,         int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,    tnlCuda,    long int, long int >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,       tnlCuda,       float, long int >( const float* data1, const float* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,      tnlCuda,      double, long int >( const double* data1, const double* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, tnlCuda, long double, long int >( const long double* data1, const long double* data2, const long int size );

extern template bool tnlArrayOperations< tnlHost >::setMemory< char,        int >( char* destination, const char& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< int,         int >( int* destination, const int& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< float,       int >( float* destination, const float& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< double,      int >( double* destination, const double& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );

#endif


#endif /* TNLARRAYOPERATIONSHOST_IMPL_H_ */
