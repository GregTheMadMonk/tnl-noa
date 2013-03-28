/***************************************************************************
                          memory_operations_impl.cu  -  description
                             -------------------
    begin                : Mar 24, 2013
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

#include <implementation/core/memory-operations.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template bool allocateMemoryHost( char*& data, const int size );
template bool allocateMemoryHost( int*& data, const int size );
template bool allocateMemoryHost( long int*& data, const int size );
template bool allocateMemoryHost( float*& data, const int size );
template bool allocateMemoryHost( double*& data, const int size );
template bool allocateMemoryHost( long double*& data, const int size );

template bool allocateMemoryHost( char*& data, const long int size );
template bool allocateMemoryHost( int*& data, const long int size );
template bool allocateMemoryHost( long int*& data, const long int size );
template bool allocateMemoryHost( float*& data, const long int size );
template bool allocateMemoryHost( double*& data, const long int size );
template bool allocateMemoryHost( long double*& data, const long int size );

template bool allocateMemoryCuda( char*& data, const int size );
template bool allocateMemoryCuda( int*& data, const int size );
template bool allocateMemoryCuda( long int*& data, const int size );
template bool allocateMemoryCuda( float*& data, const int size );
template bool allocateMemoryCuda( double*& data, const int size );
template bool allocateMemoryCuda( long double*& data, const int size );

template bool allocateMemoryCuda( char*& data, const long int size );
template bool allocateMemoryCuda( int*& data, const long int size );
template bool allocateMemoryCuda( long int*& data, const long int size );
template bool allocateMemoryCuda( float*& data, const long int size );
template bool allocateMemoryCuda( double*& data, const long int size );
template bool allocateMemoryCuda( long double*& data, const long int size );

template bool freeMemoryHost( char* data );
template bool freeMemoryHost( int* data );
template bool freeMemoryHost( long int* data );
template bool freeMemoryHost( float* data );
template bool freeMemoryHost( double* data );
template bool freeMemoryHost( long double* data );

template bool freeMemoryCuda( char* data );
template bool freeMemoryCuda( int* data );
template bool freeMemoryCuda( long int* data );
template bool freeMemoryCuda( float* data );
template bool freeMemoryCuda( double* data );
template bool freeMemoryCuda( long double* data );

template bool setMemoryHost( char* data, const char& value, const int size );
template bool setMemoryHost( int* data, const int& value, const int size );
template bool setMemoryHost( long int* data, const long int& value, const int size );
template bool setMemoryHost( float* data, const float& value, const int size );
template bool setMemoryHost( double* data, const double& value, const int size );
template bool setMemoryHost( long double* data, const long double& value, const int size );

template bool setMemoryHost( char* data, const char& value, const long int size );
template bool setMemoryHost( int* data, const int& value, const long int size );
template bool setMemoryHost( long int* data, const long int& value, const long int size );
template bool setMemoryHost( float* data, const float& value, const long int size );
template bool setMemoryHost( double* data, const double& value, const long int size );
template bool setMemoryHost( long double* data, const long double& value, const long int size );

template bool setMemoryCuda( char* data, const char& value, const int size );
template bool setMemoryCuda( int* data, const int& value, const int size );
template bool setMemoryCuda( long int* data, const long int& value, const int size );
template bool setMemoryCuda( float* data, const float& value, const int size );
template bool setMemoryCuda( double* data, const double& value, const int size );
template bool setMemoryCuda( long double* data, const long double& value, const int size );

template bool setMemoryCuda( char* data, const char& value, const long int size );
template bool setMemoryCuda( int* data, const int& value, const long int size );
template bool setMemoryCuda( long int* data, const long int& value, const long int size );
template bool setMemoryCuda( float* data, const float& value, const long int size );
template bool setMemoryCuda( double* data, const double& value, const long int size );
template bool setMemoryCuda( long double* data, const long double& value, const long int size );

template bool copyMemoryHostToHost( char* destination, const char* source, const int size );
template bool copyMemoryHostToHost( int* destination, const int* source, const int size );
template bool copyMemoryHostToHost( long int* destination, const long int* source, const int size );
template bool copyMemoryHostToHost( float* destination, const float* source, const int size );
template bool copyMemoryHostToHost( double* destination, const double* source, const int size );
template bool copyMemoryHostToHost( long double* destination, const long double* source, const int size );

template bool copyMemoryHostToHost( char* destination, const char* source, const long int size );
template bool copyMemoryHostToHost( int* destination, const int* source, const long int size );
template bool copyMemoryHostToHost( long int* destination, const long int* source, const long int size );
template bool copyMemoryHostToHost( float* destination, const float* source, const long int size );
template bool copyMemoryHostToHost( double* destination, const double* source, const long int size );
template bool copyMemoryHostToHost( long double* destination, const long double* source, const long int size );

template bool copyMemoryCudaToHost( char* destination, const char* source, const int size );
template bool copyMemoryCudaToHost( int* destination, const int* source, const int size );
template bool copyMemoryCudaToHost( long int* destination, const long int* source, const int size );
template bool copyMemoryCudaToHost( float* destination, const float* source, const int size );
template bool copyMemoryCudaToHost( double* destination, const double* source, const int size );

template bool copyMemoryCudaToHost( char* destination, const char* source, const long int size );
template bool copyMemoryCudaToHost( int* destination, const int* source, const long int size );
template bool copyMemoryCudaToHost( long int* destination, const long int* source, const long int size );
template bool copyMemoryCudaToHost( float* destination, const float* source, const long int size );
template bool copyMemoryCudaToHost( double* destination, const double* source, const long int size );

template bool copyMemoryHostToCuda( char* destination, const char* source, const int size );
template bool copyMemoryHostToCuda( int* destination, const int* source, const int size );
template bool copyMemoryHostToCuda( long int* destination, const long int* source, const int size );
template bool copyMemoryHostToCuda( float* destination, const float* source, const int size );
template bool copyMemoryHostToCuda( double* destination, const double* source, const int size );

template bool copyMemoryHostToCuda( char* destination, const char* source, const long int size );
template bool copyMemoryHostToCuda( int* destination, const int* source, const long int size );
template bool copyMemoryHostToCuda( long int* destination, const long int* source, const long int size );
template bool copyMemoryHostToCuda( float* destination, const float* source, const long int size );
template bool copyMemoryHostToCuda( double* destination, const double* source, const long int size );

template bool copyMemoryCudaToCuda( char* destination, const char* source, const int size );
template bool copyMemoryCudaToCuda( int* destination, const int* source, const int size );
template bool copyMemoryCudaToCuda( long int* destination, const long int* source, const int size );
template bool copyMemoryCudaToCuda( float* destination, const float* source, const int size );
template bool copyMemoryCudaToCuda( double* destination, const double* source, const int size );

template bool copyMemoryCudaToCuda( char* destination, const char* source, const long int size );
template bool copyMemoryCudaToCuda( int* destination, const int* source, const long int size );
template bool copyMemoryCudaToCuda( long int* destination, const long int* source, const long int size );
template bool copyMemoryCudaToCuda( float* destination, const float* source, const long int size );
template bool copyMemoryCudaToCuda( double* destination, const double* source, const long int size );

template bool compareMemoryHost( const char* data1, const char* data2, const int size );
template bool compareMemoryHost( const int* data1, const int* data2, const int size );
template bool compareMemoryHost( const long int* data1, const long int* data2, const int size );
template bool compareMemoryHost( const float* data1, const float* data2, const int size );
template bool compareMemoryHost( const double* data1, const double* data2, const int size );
template bool compareMemoryHost( const long double* data1, const long double* data2, const int size );

template bool compareMemoryHost( const char* data1, const char* data2, const long int size );
template bool compareMemoryHost( const int* data1, const int* data2, const long int size );
template bool compareMemoryHost( const long int* data1, const long int* data2, const long int size );
template bool compareMemoryHost( const float* data1, const float* data2, const long int size );
template bool compareMemoryHost( const double* data1, const double* data2, const long int size );
template bool compareMemoryHost( const long double* data1, const long double* data2, const long int size );

template bool compareMemoryHostCuda( const char* data1, const char* data2, const int size );
template bool compareMemoryHostCuda( const int* data1, const int* data2, const int size );
template bool compareMemoryHostCuda( const long int* data1, const long int* data2, const int size );
template bool compareMemoryHostCuda( const float* data1, const float* data2, const int size );
template bool compareMemoryHostCuda( const double* data1, const double* data2, const int size );
template bool compareMemoryHostCuda( const long double* data1, const long double* data2, const int size );

template bool compareMemoryHostCuda( const char* data1, const char* data2, const long int size );
template bool compareMemoryHostCuda( const int* data1, const int* data2, const long int size );
template bool compareMemoryHostCuda( const long int* data1, const long int* data2, const long int size );
template bool compareMemoryHostCuda( const float* data1, const float* data2, const long int size );
template bool compareMemoryHostCuda( const double* data1, const double* data2, const long int size );
template bool compareMemoryHostCuda( const long double* data1, const long double* data2, const long int size );

template bool compareMemoryCuda( const char* data1, const char* data2, const int size );
template bool compareMemoryCuda( const int* data1, const int* data2, const int size );
template bool compareMemoryCuda( const long int* data1, const long int* data2, const int size );
template bool compareMemoryCuda( const float* data1, const float* data2, const int size );
template bool compareMemoryCuda( const double* data1, const double* data2, const int size );
template bool compareMemoryCuda( const long double* data1, const long double* data2, const int size );

template bool compareMemoryCuda( const char* data1, const char* data2, const long int size );
template bool compareMemoryCuda( const int* data1, const int* data2, const long int size );
template bool compareMemoryCuda( const long int* data1, const long int* data2, const long int size );
template bool compareMemoryCuda( const float* data1, const float* data2, const long int size );
template bool compareMemoryCuda( const double* data1, const double* data2, const long int size );
template bool compareMemoryCuda( const long double* data1, const long double* data2, const long int size );
                                                                        
#endif                                    