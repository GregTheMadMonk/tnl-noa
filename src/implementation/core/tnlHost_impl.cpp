/***************************************************************************
                          tnlHost_impl.cpp  -  description
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

#include <core/tnlHost.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
template void tnlHost :: allocateMemory< char,        int >( char*& data, const int size );
template void tnlHost :: allocateMemory< int,         int >( int*& data, const int size );
template void tnlHost :: allocateMemory< long int,    int >( long int*& data, const int size );
template void tnlHost :: allocateMemory< float,       int >( float*& data, const int size );
template void tnlHost :: allocateMemory< double,      int >( double*& data, const int size );
template void tnlHost :: allocateMemory< long double, int >( long double*& data, const int size );
template void tnlHost :: allocateMemory< char,        long int >( char*& data, const long int size );
template void tnlHost :: allocateMemory< int,         long int >( int*& data, const long int size );
template void tnlHost :: allocateMemory< long int,    long int >( long int*& data, const long int size );
template void tnlHost :: allocateMemory< float,       long int >( float*& data, const long int size );
template void tnlHost :: allocateMemory< double,      long int >( double*& data, const long int size );
template void tnlHost :: allocateMemory< long double, long int >( long double*& data, const long int size );

template void tnlHost :: freeMemory< char        >( char* data );
template void tnlHost :: freeMemory< int         >( int* data );
template void tnlHost :: freeMemory< long int    >( long int* data );
template void tnlHost :: freeMemory< float       >( float* data );
template void tnlHost :: freeMemory< double      >( double* data );
template void tnlHost :: freeMemory< long double >( long double* data );

template void tnlHost :: setMemoryElement< char        >( char* data, const char& value );
template void tnlHost :: setMemoryElement< int         >( int* data, const int& value );
template void tnlHost :: setMemoryElement< long int    >( long int* data, const long int& value );
template void tnlHost :: setMemoryElement< float       >( float* data, const float& value );
template void tnlHost :: setMemoryElement< double      >( double* data, const double& value );
template void tnlHost :: setMemoryElement< long double >( long double* data, const long double& value );

template char        tnlHost :: getMemoryElement< char        >( char* data );
template int         tnlHost :: getMemoryElement< int         >( int* data );
template long int    tnlHost :: getMemoryElement< long int    >( long int* data );
template float       tnlHost :: getMemoryElement< float       >( float* data );
template double      tnlHost :: getMemoryElement< double      >( double* data );
template long double tnlHost :: getMemoryElement< long double >( long double* data );

template char&        tnlHost :: getArrayElementReference< char,        int >( char* data, const int i );
template int&         tnlHost :: getArrayElementReference< int,         int >( int* data, const int i );
template long int&    tnlHost :: getArrayElementReference< long int,    int >( long int* data, const int i );
template float&       tnlHost :: getArrayElementReference< float,       int >( float* data, const int i );
template double&      tnlHost :: getArrayElementReference< double,      int >( double* data, const int i );
template long double& tnlHost :: getArrayElementReference< long double, int >( long double* data, const int i );

template char&        tnlHost :: getArrayElementReference< char,        long int >( char* data, const long int i );
template int&         tnlHost :: getArrayElementReference< int,         long int >( int* data, const long int i );
template long int&    tnlHost :: getArrayElementReference< long int,    long int >( long int* data, const long int i );
template float&       tnlHost :: getArrayElementReference< float,       long int >( float* data, const long int i );
template double&      tnlHost :: getArrayElementReference< double,      long int >( double* data, const long int i );
template long double& tnlHost :: getArrayElementReference< long double, long int >( long double* data, const long int i );

template const char&        tnlHost :: getArrayElementReference< char,        int >( const char* data, const int i );
template const int&         tnlHost :: getArrayElementReference< int,         int >( const int* data, const int i );
template const long int&    tnlHost :: getArrayElementReference< long int,    int >( const long int* data, const int i );
template const float&       tnlHost :: getArrayElementReference< float,       int >( const float* data, const int i );
template const double&      tnlHost :: getArrayElementReference< double,      int >( const double* data, const int i );
template const long double& tnlHost :: getArrayElementReference< long double, int >( const long double* data, const int i );

template const char&        tnlHost :: getArrayElementReference< char,        long int >( const char* data, const long int i );
template const int&         tnlHost :: getArrayElementReference< int,         long int >( const int* data, const long int i );
template const long int&    tnlHost :: getArrayElementReference< long int,    long int >( const long int* data, const long int i );
template const float&       tnlHost :: getArrayElementReference< float,       long int >( const float* data, const long int i );
template const double&      tnlHost :: getArrayElementReference< double,      long int >( const double* data, const long int i );
template const long double& tnlHost :: getArrayElementReference< long double, long int >( const long double* data, const long int i );

template bool tnlHost :: memcpy< char,        int, tnlHost >( char* destination, const char* source, const int size );
template bool tnlHost :: memcpy< int,         int, tnlHost >( int* destination, const int* source, const int size );
template bool tnlHost :: memcpy< long int,    int, tnlHost >( long int* destination, const long int* source, const int size );
template bool tnlHost :: memcpy< float,       int, tnlHost >( float* destination, const float* source, const int size );
template bool tnlHost :: memcpy< double,      int, tnlHost >( double* destination, const double* source, const int size );
template bool tnlHost :: memcpy< long double, int, tnlHost >( long double* destination, const long double* source, const int size );
template bool tnlHost :: memcpy< char,        long int, tnlHost >( char* destination, const char* source, const long int size );
template bool tnlHost :: memcpy< int,         long int, tnlHost >( int* destination, const int* source, const long int size );
template bool tnlHost :: memcpy< long int,    long int, tnlHost >( long int* destination, const long int* source, const long int size );
template bool tnlHost :: memcpy< float,       long int, tnlHost >( float* destination, const float* source, const long int size );
template bool tnlHost :: memcpy< double,      long int, tnlHost >( double* destination, const double* source, const long int size );
template bool tnlHost :: memcpy< long double, long int, tnlHost >( long double* destination, const long double* source, const long int size );
template bool tnlHost :: memcpy< char,        int, tnlCuda >( char* destination, const char* source, const int size );
template bool tnlHost :: memcpy< int,         int, tnlCuda >( int* destination, const int* source, const int size );
template bool tnlHost :: memcpy< long int,    int, tnlCuda >( long int* destination, const long int* source, const int size );
template bool tnlHost :: memcpy< float,       int, tnlCuda >( float* destination, const float* source, const int size );
template bool tnlHost :: memcpy< double,      int, tnlCuda >( double* destination, const double* source, const int size );
template bool tnlHost :: memcpy< long double, int, tnlCuda >( long double* destination, const long double* source, const int size );
template bool tnlHost :: memcpy< char,        long int, tnlCuda >( char* destination, const char* source, const long int size );
template bool tnlHost :: memcpy< int,         long int, tnlCuda >( int* destination, const int* source, const long int size );
template bool tnlHost :: memcpy< long int,    long int, tnlCuda >( long int* destination, const long int* source, const long int size );
template bool tnlHost :: memcpy< float,       long int, tnlCuda >( float* destination, const float* source, const long int size );
template bool tnlHost :: memcpy< double,      long int, tnlCuda >( double* destination, const double* source, const long int size );
template bool tnlHost :: memcpy< long double, long int, tnlCuda >( long double* destination, const long double* source, const long int size );

template bool tnlHost :: memcmp< char,        int, tnlHost >( const char* data1, const char* data2, const int size );
template bool tnlHost :: memcmp< int,         int, tnlHost >( const int* data1, const int* data2, const int size );
template bool tnlHost :: memcmp< long int,    int, tnlHost >( const long int* data1, const long int* data2, const int size );
template bool tnlHost :: memcmp< float,       int, tnlHost >( const float* data1, const float* data2, const int size );
template bool tnlHost :: memcmp< double,      int, tnlHost >( const double* data1, const double* data2, const int size );
template bool tnlHost :: memcmp< long double, int, tnlHost >( const long double* data1, const long double* data2, const int size );
template bool tnlHost :: memcmp< char,        long int, tnlHost >( const char* data1, const char* data2, const long int size );
template bool tnlHost :: memcmp< int,         long int, tnlHost >( const int* data1, const int* data2, const long int size );
template bool tnlHost :: memcmp< long int,    long int, tnlHost >( const long int* data1, const long int* data2, const long int size );
template bool tnlHost :: memcmp< float,       long int, tnlHost >( const float* data1, const float* data2, const long int size );
template bool tnlHost :: memcmp< double,      long int, tnlHost >( const double* data1, const double* data2, const long int size );
template bool tnlHost :: memcmp< long double, long int, tnlHost >( const long double* data1, const long double* data2, const long int size );
template bool tnlHost :: memcmp< char,        int, tnlCuda >( const char* data1, const char* data2, const int size );
template bool tnlHost :: memcmp< int,         int, tnlCuda >( const int* data1, const int* data2, const int size );
template bool tnlHost :: memcmp< long int,    int, tnlCuda >( const long int* data1, const long int* data2, const int size );
template bool tnlHost :: memcmp< float,       int, tnlCuda >( const float* data1, const float* data2, const int size );
template bool tnlHost :: memcmp< double,      int, tnlCuda >( const double* data1, const double* data2, const int size );
template bool tnlHost :: memcmp< long double, int, tnlCuda >( const long double* data1, const long double* data2, const int size );
template bool tnlHost :: memcmp< char,        long int, tnlCuda >( const char* data1, const char* data2, const long int size );
template bool tnlHost :: memcmp< int,         long int, tnlCuda >( const int* data1, const int* data2, const long int size );
template bool tnlHost :: memcmp< long int,    long int, tnlCuda >( const long int* data1, const long int* data2, const long int size );
template bool tnlHost :: memcmp< float,       long int, tnlCuda >( const float* data1, const float* data2, const long int size );
template bool tnlHost :: memcmp< double,      long int, tnlCuda >( const double* data1, const double* data2, const long int size );
template bool tnlHost :: memcmp< long double, long int, tnlCuda >( const long double* data1, const long double* data2, const long int size );

template bool tnlHost :: memset< char,        int >( char* destination, const char& value, const int size );
template bool tnlHost :: memset< int,         int >( int* destination, const int& value, const int size );
template bool tnlHost :: memset< long int,    int >( long int* destination, const long int& value, const int size );
template bool tnlHost :: memset< float,       int >( float* destination, const float& value, const int size );
template bool tnlHost :: memset< double,      int >( double* destination, const double& value, const int size );
template bool tnlHost :: memset< long double, int >( long double* destination, const long double& value, const int size );
template bool tnlHost :: memset< char,        long int >( char* destination, const char& value, const long int size );
template bool tnlHost :: memset< int,         long int >( int* destination, const int& value, const long int size );
template bool tnlHost :: memset< long int,    long int >( long int* destination, const long int& value, const long int size );
template bool tnlHost :: memset< float,       long int >( float* destination, const float& value, const long int size );
template bool tnlHost :: memset< double,      long int >( double* destination, const double& value, const long int size );
template bool tnlHost :: memset< long double, long int >( long double* destination, const long double& value, const long int size );
#endif

