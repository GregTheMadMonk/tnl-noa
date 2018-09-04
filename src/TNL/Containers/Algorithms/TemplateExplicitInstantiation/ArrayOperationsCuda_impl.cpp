/***************************************************************************
                          ArrayOperationsCuda_impl.cpp  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {    
namespace Algorithms {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template bool ArrayOperations< Devices::Cuda >::allocateMemory< char,        int >( char*& data, const int size );
template bool ArrayOperations< Devices::Cuda >::allocateMemory< int,         int >( int*& data, const int size );
template bool ArrayOperations< Devices::Cuda >::allocateMemory< long int,    int >( long int*& data, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::allocateMemory< float,       int >( float*& data, const int size );
#endif
template bool ArrayOperations< Devices::Cuda >::allocateMemory< double,      int >( double*& data, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::allocateMemory< long double, int >( long double*& data, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda >::allocateMemory< char,        long int >( char*& data, const long int size );
template bool ArrayOperations< Devices::Cuda >::allocateMemory< int,         long int >( int*& data, const long int size );
template bool ArrayOperations< Devices::Cuda >::allocateMemory< long int,    long int >( long int*& data, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::allocateMemory< float,       long int >( float*& data, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda >::allocateMemory< double,      long int >( double*& data, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::allocateMemory< long double, long int >( long double*& data, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Cuda >::freeMemory< char        >( char* data );
template bool ArrayOperations< Devices::Cuda >::freeMemory< int         >( int* data );
template bool ArrayOperations< Devices::Cuda >::freeMemory< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::freeMemory< float       >( float* data );
#endif
template bool ArrayOperations< Devices::Cuda >::freeMemory< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::freeMemory< long double >( long double* data );
#endif

template void ArrayOperations< Devices::Cuda >::setMemoryElement< char        >( char* data, const char& value );
template void ArrayOperations< Devices::Cuda >::setMemoryElement< int         >( int* data, const int& value );
template void ArrayOperations< Devices::Cuda >::setMemoryElement< long int    >( long int* data, const long int& value );
#ifdef INSTANTIATE_FLOAT
template void ArrayOperations< Devices::Cuda >::setMemoryElement< float       >( float* data, const float& value );
#endif
template void ArrayOperations< Devices::Cuda >::setMemoryElement< double      >( double* data, const double& value );
#ifdef INSTANTIATE_LONG_DOUBLE
template void ArrayOperations< Devices::Cuda >::setMemoryElement< long double >( long double* data, const long double& value );
#endif

template char        ArrayOperations< Devices::Cuda >::getMemoryElement< char        >( const char* data );
template int         ArrayOperations< Devices::Cuda >::getMemoryElement< int         >( const int* data );
template long int    ArrayOperations< Devices::Cuda >::getMemoryElement< long int    >( const long int* data );
#ifdef INSTANTIATE_FLOAT
template float       ArrayOperations< Devices::Cuda >::getMemoryElement< float       >( const float* data );
#endif
template double      ArrayOperations< Devices::Cuda >::getMemoryElement< double      >( const double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double ArrayOperations< Devices::Cuda >::getMemoryElement< long double >( const long double* data );
#endif

template bool ArrayOperations< Devices::Cuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
template bool ArrayOperations< Devices::Cuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
template bool ArrayOperations< Devices::Cuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
template bool ArrayOperations< Devices::Cuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
template bool ArrayOperations< Devices::Cuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
template bool ArrayOperations< Devices::Cuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Cuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
template bool ArrayOperations< Devices::Cuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
template bool ArrayOperations< Devices::Cuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
template bool ArrayOperations< Devices::Cuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
template bool ArrayOperations< Devices::Cuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
template bool ArrayOperations< Devices::Cuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Cuda >::setMemory< char,        int >( char* destination, const char& value, const int size );
template bool ArrayOperations< Devices::Cuda >::setMemory< int,         int >( int* destination, const int& value, const int size );
template bool ArrayOperations< Devices::Cuda >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::setMemory< float,       int >( float* destination, const float& value, const int size );
#endif
template bool ArrayOperations< Devices::Cuda >::setMemory< double,      int >( double* destination, const double& value, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Cuda >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
template bool ArrayOperations< Devices::Cuda >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
template bool ArrayOperations< Devices::Cuda >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Cuda >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
#endif
template bool ArrayOperations< Devices::Cuda >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Cuda >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );
#endif
#endif

#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
