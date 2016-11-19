/***************************************************************************
                          ArrayOperationsHost_impl.cpp  -  description
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

template bool ArrayOperations< Devices::Host >::allocateMemory< char,        int >( char*& data, const int size );
template bool ArrayOperations< Devices::Host >::allocateMemory< int,         int >( int*& data, const int size );
template bool ArrayOperations< Devices::Host >::allocateMemory< long int,    int >( long int*& data, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::allocateMemory< float,       int >( float*& data, const int size );
#endif
template bool ArrayOperations< Devices::Host >::allocateMemory< double,      int >( double*& data, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::allocateMemory< long double, int >( long double*& data, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host >::allocateMemory< char,        long int >( char*& data, const long int size );
template bool ArrayOperations< Devices::Host >::allocateMemory< int,         long int >( int*& data, const long int size );
template bool ArrayOperations< Devices::Host >::allocateMemory< long int,    long int >( long int*& data, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::allocateMemory< float,       long int >( float*& data, const long int size );
#endif
template bool ArrayOperations< Devices::Host >::allocateMemory< double,      long int >( double*& data, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::allocateMemory< long double, long int >( long double*& data, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Host >::freeMemory< char        >( char* data );
template bool ArrayOperations< Devices::Host >::freeMemory< int         >( int* data );
template bool ArrayOperations< Devices::Host >::freeMemory< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::freeMemory< float       >( float* data );
#endif
template bool ArrayOperations< Devices::Host >::freeMemory< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::freeMemory< long double >( long double* data );
#endif

template void ArrayOperations< Devices::Host >::setMemoryElement< char        >( char* data, const char& value );
template void ArrayOperations< Devices::Host >::setMemoryElement< int         >( int* data, const int& value );
template void ArrayOperations< Devices::Host >::setMemoryElement< long int    >( long int* data, const long int& value );
#ifdef INSTANTIATE_FLOAT
template void ArrayOperations< Devices::Host >::setMemoryElement< float       >( float* data, const float& value );
#endif
template void ArrayOperations< Devices::Host >::setMemoryElement< double      >( double* data, const double& value );
#ifdef INSTANTIATE_LONG_DOUBLE
template void ArrayOperations< Devices::Host >::setMemoryElement< long double >( long double* data, const long double& value );
#endif

template char        ArrayOperations< Devices::Host >::getMemoryElement< char        >( char* data );
template int         ArrayOperations< Devices::Host >::getMemoryElement< int         >( int* data );
template long int    ArrayOperations< Devices::Host >::getMemoryElement< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
template float       ArrayOperations< Devices::Host >::getMemoryElement< float       >( float* data );
#endif
template double      ArrayOperations< Devices::Host >::getMemoryElement< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double ArrayOperations< Devices::Host >::getMemoryElement< long double >( long double* data );
#endif

template char&        ArrayOperations< Devices::Host >::getArrayElementReference< char,        int >( char* data, const int i );
template int&         ArrayOperations< Devices::Host >::getArrayElementReference< int,         int >( int* data, const int i );
template long int&    ArrayOperations< Devices::Host >::getArrayElementReference< long int,    int >( long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
template float&       ArrayOperations< Devices::Host >::getArrayElementReference< float,       int >( float* data, const int i );
#endif
template double&      ArrayOperations< Devices::Host >::getArrayElementReference< double,      int >( double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double& ArrayOperations< Devices::Host >::getArrayElementReference< long double, int >( long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
template char&        ArrayOperations< Devices::Host >::getArrayElementReference< char,        long int >( char* data, const long int i );
template int&         ArrayOperations< Devices::Host >::getArrayElementReference< int,         long int >( int* data, const long int i );
template long int&    ArrayOperations< Devices::Host >::getArrayElementReference< long int,    long int >( long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
template float&       ArrayOperations< Devices::Host >::getArrayElementReference< float,       long int >( float* data, const long int i );
#endif
template double&      ArrayOperations< Devices::Host >::getArrayElementReference< double,      long int >( double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double& ArrayOperations< Devices::Host >::getArrayElementReference< long double, long int >( long double* data, const long int i );
#endif
#endif

template const char&        ArrayOperations< Devices::Host >::getArrayElementReference< char,        int >( const char* data, const int i );
template const int&         ArrayOperations< Devices::Host >::getArrayElementReference< int,         int >( const int* data, const int i );
template const long int&    ArrayOperations< Devices::Host >::getArrayElementReference< long int,    int >( const long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
template const float&       ArrayOperations< Devices::Host >::getArrayElementReference< float,       int >( const float* data, const int i );
#endif
template const double&      ArrayOperations< Devices::Host >::getArrayElementReference< double,      int >( const double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
template const long double& ArrayOperations< Devices::Host >::getArrayElementReference< long double, int >( const long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
template const char&        ArrayOperations< Devices::Host >::getArrayElementReference< char,        long int >( const char* data, const long int i );
template const int&         ArrayOperations< Devices::Host >::getArrayElementReference< int,         long int >( const int* data, const long int i );
template const long int&    ArrayOperations< Devices::Host >::getArrayElementReference< long int,    long int >( const long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
template const float&       ArrayOperations< Devices::Host >::getArrayElementReference< float,       long int >( const float* data, const long int i );
#endif
template const double&      ArrayOperations< Devices::Host >::getArrayElementReference< double,      long int >( const double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
template const long double& ArrayOperations< Devices::Host >::getArrayElementReference< long double, long int >( const long double* data, const long int i );
#endif
#endif

template bool ArrayOperations< Devices::Host >::copyMemory< char,               char, int >( char* destination, const char* source, const int size );
template bool ArrayOperations< Devices::Host >::copyMemory< int,                 int, int >( int* destination, const int* source, const int size );
template bool ArrayOperations< Devices::Host >::copyMemory< long int,       long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::copyMemory< float,             float, int >( float* destination, const float* source, const int size );
#endif
template bool ArrayOperations< Devices::Host >::copyMemory< double,           double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::copyMemory< long double, long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host >::copyMemory< char,               char, long int >( char* destination, const char* source, const long int size );
template bool ArrayOperations< Devices::Host >::copyMemory< int,                 int, long int >( int* destination, const int* source, const long int size );
template bool ArrayOperations< Devices::Host >::copyMemory< long int,       long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::copyMemory< float,             float, long int >( float* destination, const float* source, const long int size );
#endif
template bool ArrayOperations< Devices::Host >::copyMemory< double,           double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::copyMemory< long double, long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Host >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
template bool ArrayOperations< Devices::Host >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
template bool ArrayOperations< Devices::Host >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
template bool ArrayOperations< Devices::Host >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
template bool ArrayOperations< Devices::Host >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
template bool ArrayOperations< Devices::Host >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
template bool ArrayOperations< Devices::Host >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

template bool ArrayOperations< Devices::Host >::setMemory< char,        int >( char* destination, const char& value, const int size );
template bool ArrayOperations< Devices::Host >::setMemory< int,         int >( int* destination, const int& value, const int size );
template bool ArrayOperations< Devices::Host >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::setMemory< float,       int >( float* destination, const float& value, const int size );
#endif
template bool ArrayOperations< Devices::Host >::setMemory< double,      int >( double* destination, const double& value, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
template bool ArrayOperations< Devices::Host >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
template bool ArrayOperations< Devices::Host >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
template bool ArrayOperations< Devices::Host >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
#ifdef INSTANTIATE_FLOAT
template bool ArrayOperations< Devices::Host >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
#endif
template bool ArrayOperations< Devices::Host >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
template bool ArrayOperations< Devices::Host >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );
#endif
#endif

#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
