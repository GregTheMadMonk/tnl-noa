/***************************************************************************
                          tnlHost_impl.h  -  description
                             -------------------
    begin                : Nov 7, 2012
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

#ifndef TNLHOST_IMPL_H_
#define TNLHOST_IMPL_H_

#include <core/tnlCuda.h>

inline tnlString tnlHost :: getDeviceType()
{
   return tnlString( "tnlHost" );
};

inline tnlDeviceEnum tnlHost :: getDevice()
{
   return tnlHostDevice;
};

template< typename Element, typename Index >
void tnlHost :: allocateMemory( Element*& data, const Index size )
{
   allocateMemoryHost( data, size );
};

template< typename Element >
void tnlHost :: freeMemory( Element* data )
{
   freeMemoryHost( data );
};

template< typename Element >
void tnlHost :: setMemoryElement( Element* data,
                                         const Element& value )
{
   *data = value;
};

template< typename Element >
Element tnlHost :: getMemoryElement( Element* data )
{
   return *data;
};

template< typename Element, typename Index >
Element& tnlHost :: getArrayElementReference( Element* data, const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index >
const Element& tnlHost :: getArrayElementReference(const Element* data, const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index, typename Device >
bool tnlHost :: memcpy( Element* destination,
                               const Element* source,
                               const Index size )
{
   switch( Device :: getDevice() )
   {
      case tnlHostDevice:
         return copyMemoryHostToHost( destination, source, size );
      case tnlCudaDevice:
         return copyMemoryCudaToHost( destination, source, size );
   }
   return true;
};

template< typename Element, typename Index, typename Device >
bool tnlHost :: memcmp( const Element* data1,
                               const Element* data2,
                               const Index size )
{
   switch( Device :: getDevice() )
   {
      case tnlHostDevice:
         return compareMemoryHost( data1, data2, size );
      case tnlCudaDevice:
         return compareMemoryHostCuda( data1, data2, size );
   }
};

template< typename Element, typename Index >
bool tnlHost :: memset( Element* destination,
                               const Element& value,
                               const Index size )
{
   return setMemoryHost( destination, value, size );
};

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template void tnlHost :: allocateMemory< char,        int >( char*& data, const int size );
extern template void tnlHost :: allocateMemory< int,         int >( int*& data, const int size );
extern template void tnlHost :: allocateMemory< long int,    int >( long int*& data, const int size );
extern template void tnlHost :: allocateMemory< float,       int >( float*& data, const int size );
extern template void tnlHost :: allocateMemory< double,      int >( double*& data, const int size );
extern template void tnlHost :: allocateMemory< long double, int >( long double*& data, const int size );
extern template void tnlHost :: allocateMemory< char,        long int >( char*& data, const long int size );
extern template void tnlHost :: allocateMemory< int,         long int >( int*& data, const long int size );
extern template void tnlHost :: allocateMemory< long int,    long int >( long int*& data, const long int size );
extern template void tnlHost :: allocateMemory< float,       long int >( float*& data, const long int size );
extern template void tnlHost :: allocateMemory< double,      long int >( double*& data, const long int size );
extern template void tnlHost :: allocateMemory< long double, long int >( long double*& data, const long int size );

extern template void tnlHost :: freeMemory< char        >( char* data );
extern template void tnlHost :: freeMemory< int         >( int* data );
extern template void tnlHost :: freeMemory< long int    >( long int* data );
extern template void tnlHost :: freeMemory< float       >( float* data );
extern template void tnlHost :: freeMemory< double      >( double* data );
extern template void tnlHost :: freeMemory< long double >( long double* data );

extern template void tnlHost :: setMemoryElement< char        >( char* data, const char& value );
extern template void tnlHost :: setMemoryElement< int         >( int* data, const int& value );
extern template void tnlHost :: setMemoryElement< long int    >( long int* data, const long int& value );
extern template void tnlHost :: setMemoryElement< float       >( float* data, const float& value );
extern template void tnlHost :: setMemoryElement< double      >( double* data, const double& value );
extern template void tnlHost :: setMemoryElement< long double >( long double* data, const long double& value );

extern template char        tnlHost :: getMemoryElement< char        >( char* data );
extern template int         tnlHost :: getMemoryElement< int         >( int* data );
extern template long int    tnlHost :: getMemoryElement< long int    >( long int* data );
extern template float       tnlHost :: getMemoryElement< float       >( float* data );
extern template double      tnlHost :: getMemoryElement< double      >( double* data );
extern template long double tnlHost :: getMemoryElement< long double >( long double* data );

extern template char&        tnlHost :: getArrayElementReference< char,        int >( char* data, const int i );
extern template int&         tnlHost :: getArrayElementReference< int,         int >( int* data, const int i );
extern template long int&    tnlHost :: getArrayElementReference< long int,    int >( long int* data, const int i );
extern template float&       tnlHost :: getArrayElementReference< float,       int >( float* data, const int i );
extern template double&      tnlHost :: getArrayElementReference< double,      int >( double* data, const int i );
extern template long double& tnlHost :: getArrayElementReference< long double, int >( long double* data, const int i );

extern template char&        tnlHost :: getArrayElementReference< char,        long int >( char* data, const long int i );
extern template int&         tnlHost :: getArrayElementReference< int,         long int >( int* data, const long int i );
extern template long int&    tnlHost :: getArrayElementReference< long int,    long int >( long int* data, const long int i );
extern template float&       tnlHost :: getArrayElementReference< float,       long int >( float* data, const long int i );
extern template double&      tnlHost :: getArrayElementReference< double,      long int >( double* data, const long int i );
extern template long double& tnlHost :: getArrayElementReference< long double, long int >( long double* data, const long int i );

extern template const char&        tnlHost :: getArrayElementReference< char,        int >( const char* data, const int i );
extern template const int&         tnlHost :: getArrayElementReference< int,         int >( const int* data, const int i );
extern template const long int&    tnlHost :: getArrayElementReference< long int,    int >( const long int* data, const int i );
extern template const float&       tnlHost :: getArrayElementReference< float,       int >( const float* data, const int i );
extern template const double&      tnlHost :: getArrayElementReference< double,      int >( const double* data, const int i );
extern template const long double& tnlHost :: getArrayElementReference< long double, int >( const long double* data, const int i );

extern template const char&        tnlHost :: getArrayElementReference< char,        long int >( const char* data, const long int i );
extern template const int&         tnlHost :: getArrayElementReference< int,         long int >( const int* data, const long int i );
extern template const long int&    tnlHost :: getArrayElementReference< long int,    long int >( const long int* data, const long int i );
extern template const float&       tnlHost :: getArrayElementReference< float,       long int >( const float* data, const long int i );
extern template const double&      tnlHost :: getArrayElementReference< double,      long int >( const double* data, const long int i );
extern template const long double& tnlHost :: getArrayElementReference< long double, long int >( const long double* data, const long int i );

extern template bool tnlHost :: memcpy< char,        int, tnlHost >( char* destination, const char* source, const int size );
extern template bool tnlHost :: memcpy< int,         int, tnlHost >( int* destination, const int* source, const int size );
extern template bool tnlHost :: memcpy< long int,    int, tnlHost >( long int* destination, const long int* source, const int size );
extern template bool tnlHost :: memcpy< float,       int, tnlHost >( float* destination, const float* source, const int size );
extern template bool tnlHost :: memcpy< double,      int, tnlHost >( double* destination, const double* source, const int size );
extern template bool tnlHost :: memcpy< long double, int, tnlHost >( long double* destination, const long double* source, const int size );
extern template bool tnlHost :: memcpy< char,        long int, tnlHost >( char* destination, const char* source, const long int size );
extern template bool tnlHost :: memcpy< int,         long int, tnlHost >( int* destination, const int* source, const long int size );
extern template bool tnlHost :: memcpy< long int,    long int, tnlHost >( long int* destination, const long int* source, const long int size );
extern template bool tnlHost :: memcpy< float,       long int, tnlHost >( float* destination, const float* source, const long int size );
extern template bool tnlHost :: memcpy< double,      long int, tnlHost >( double* destination, const double* source, const long int size );
extern template bool tnlHost :: memcpy< long double, long int, tnlHost >( long double* destination, const long double* source, const long int size );
extern template bool tnlHost :: memcpy< char,        int, tnlCuda >( char* destination, const char* source, const int size );
extern template bool tnlHost :: memcpy< int,         int, tnlCuda >( int* destination, const int* source, const int size );
extern template bool tnlHost :: memcpy< long int,    int, tnlCuda >( long int* destination, const long int* source, const int size );
extern template bool tnlHost :: memcpy< float,       int, tnlCuda >( float* destination, const float* source, const int size );
extern template bool tnlHost :: memcpy< double,      int, tnlCuda >( double* destination, const double* source, const int size );
extern template bool tnlHost :: memcpy< long double, int, tnlCuda >( long double* destination, const long double* source, const int size );
extern template bool tnlHost :: memcpy< char,        long int, tnlCuda >( char* destination, const char* source, const long int size );
extern template bool tnlHost :: memcpy< int,         long int, tnlCuda >( int* destination, const int* source, const long int size );
extern template bool tnlHost :: memcpy< long int,    long int, tnlCuda >( long int* destination, const long int* source, const long int size );
extern template bool tnlHost :: memcpy< float,       long int, tnlCuda >( float* destination, const float* source, const long int size );
extern template bool tnlHost :: memcpy< double,      long int, tnlCuda >( double* destination, const double* source, const long int size );
extern template bool tnlHost :: memcpy< long double, long int, tnlCuda >( long double* destination, const long double* source, const long int size );

extern template bool tnlHost :: memcmp< char,        int, tnlHost >( const char* data1, const char* data2, const int size );
extern template bool tnlHost :: memcmp< int,         int, tnlHost >( const int* data1, const int* data2, const int size );
extern template bool tnlHost :: memcmp< long int,    int, tnlHost >( const long int* data1, const long int* data2, const int size );
extern template bool tnlHost :: memcmp< float,       int, tnlHost >( const float* data1, const float* data2, const int size );
extern template bool tnlHost :: memcmp< double,      int, tnlHost >( const double* data1, const double* data2, const int size );
extern template bool tnlHost :: memcmp< long double, int, tnlHost >( const long double* data1, const long double* data2, const int size );
extern template bool tnlHost :: memcmp< char,        long int, tnlHost >( const char* data1, const char* data2, const long int size );
extern template bool tnlHost :: memcmp< int,         long int, tnlHost >( const int* data1, const int* data2, const long int size );
extern template bool tnlHost :: memcmp< long int,    long int, tnlHost >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlHost :: memcmp< float,       long int, tnlHost >( const float* data1, const float* data2, const long int size );
extern template bool tnlHost :: memcmp< double,      long int, tnlHost >( const double* data1, const double* data2, const long int size );
extern template bool tnlHost :: memcmp< long double, long int, tnlHost >( const long double* data1, const long double* data2, const long int size );
extern template bool tnlHost :: memcmp< char,        int, tnlCuda >( const char* data1, const char* data2, const int size );
extern template bool tnlHost :: memcmp< int,         int, tnlCuda >( const int* data1, const int* data2, const int size );
extern template bool tnlHost :: memcmp< long int,    int, tnlCuda >( const long int* data1, const long int* data2, const int size );
extern template bool tnlHost :: memcmp< float,       int, tnlCuda >( const float* data1, const float* data2, const int size );
extern template bool tnlHost :: memcmp< double,      int, tnlCuda >( const double* data1, const double* data2, const int size );
extern template bool tnlHost :: memcmp< long double, int, tnlCuda >( const long double* data1, const long double* data2, const int size );
extern template bool tnlHost :: memcmp< char,        long int, tnlCuda >( const char* data1, const char* data2, const long int size );
extern template bool tnlHost :: memcmp< int,         long int, tnlCuda >( const int* data1, const int* data2, const long int size );
extern template bool tnlHost :: memcmp< long int,    long int, tnlCuda >( const long int* data1, const long int* data2, const long int size );
extern template bool tnlHost :: memcmp< float,       long int, tnlCuda >( const float* data1, const float* data2, const long int size );
extern template bool tnlHost :: memcmp< double,      long int, tnlCuda >( const double* data1, const double* data2, const long int size );
extern template bool tnlHost :: memcmp< long double, long int, tnlCuda >( const long double* data1, const long double* data2, const long int size );

extern template bool tnlHost :: memset< char,        int >( char* destination, const char& value, const int size );
extern template bool tnlHost :: memset< int,         int >( int* destination, const int& value, const int size );
extern template bool tnlHost :: memset< long int,    int >( long int* destination, const long int& value, const int size );
extern template bool tnlHost :: memset< float,       int >( float* destination, const float& value, const int size );
extern template bool tnlHost :: memset< double,      int >( double* destination, const double& value, const int size );
extern template bool tnlHost :: memset< long double, int >( long double* destination, const long double& value, const int size );
extern template bool tnlHost :: memset< char,        long int >( char* destination, const char& value, const long int size );
extern template bool tnlHost :: memset< int,         long int >( int* destination, const int& value, const long int size );
extern template bool tnlHost :: memset< long int,    long int >( long int* destination, const long int& value, const long int size );
extern template bool tnlHost :: memset< float,       long int >( float* destination, const float& value, const long int size );
extern template bool tnlHost :: memset< double,      long int >( double* destination, const double& value, const long int size );
extern template bool tnlHost :: memset< long double, long int >( long double* destination, const long double& value, const long int size );

#endif

#endif /* TNLHOST_IMPL_H_ */
