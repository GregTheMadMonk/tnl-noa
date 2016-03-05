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

#include <type_traits>
#include <tnlConfig.h>
#include <string.h>

template< typename Element, typename Index >
bool tnlArrayOperations< tnlHost >::allocateMemory( Element*& data,
                                                      const Index size )
{
   if( ! ( data = new Element[ size ] ) )
      return false;
   return true;
}

template< typename Element >
bool tnlArrayOperations< tnlHost >::freeMemory( Element* data )
{
   delete[] data;
   return true;
}
template< typename Element >
void tnlArrayOperations< tnlHost >::setMemoryElement( Element* data,
                                                        const Element& value )
{
   *data = value;
};

template< typename Element >
Element tnlArrayOperations< tnlHost >::getMemoryElement( Element* data )
{
   return *data;
};

template< typename Element, typename Index >
Element& tnlArrayOperations< tnlHost >::getArrayElementReference( Element* data,
                                                                  const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index >
const Element& tnlArrayOperations< tnlHost >::getArrayElementReference( const Element* data,
                                                                       const Index i )
{
   return data[ i ];
};

template< typename Element, typename Index >
bool tnlArrayOperations< tnlHost >::setMemory( Element* data,
                                               const Element& value,
                                               const Index size )
{
   for( Index i = 0; i < size; i ++ )
      data[ i ] = value;
   return true;
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlHost >::copyMemory( DestinationElement* destination,
                                                const SourceElement* source,
                                                const Index size )
{
   if( std::is_same< DestinationElement, SourceElement >::value )
      memcpy( destination, source, size * sizeof( DestinationElement ) );
   else
      for( Index i = 0; i < size; i ++ )
         destination[ i ] = ( DestinationElement ) source[ i ];
   return true;
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool tnlArrayOperations< tnlHost >::compareMemory( const DestinationElement* destination,
                                                   const SourceElement* source,
                                                   const Index size )
{
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      if( memcmp( destination, source, size * sizeof( DestinationElement ) ) != 0 )
         return false;
   }
   else
      for( Index i = 0; i < size; i ++ )
         if( ! ( destination[ i ] == source[ i ] ) )
            return false;
   return true;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template bool tnlArrayOperations< tnlHost >::allocateMemory< char,        int >( char*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< int,         int >( int*& data, const int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long int,    int >( long int*& data, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< float,       int >( float*& data, const int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< double,      int >( double*& data, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long double, int >( long double*& data, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< char,        long int >( char*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< int,         long int >( int*& data, const long int size );
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long int,    long int >( long int*& data, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< float,       long int >( float*& data, const long int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< double,      long int >( double*& data, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::allocateMemory< long double, long int >( long double*& data, const long int size );
#endif
#endif

extern template bool tnlArrayOperations< tnlHost >::freeMemory< char        >( char* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< int         >( int* data );
extern template bool tnlArrayOperations< tnlHost >::freeMemory< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::freeMemory< float       >( float* data );
#endif
extern template bool tnlArrayOperations< tnlHost >::freeMemory< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::freeMemory< long double >( long double* data );
#endif

extern template void tnlArrayOperations< tnlHost >::setMemoryElement< char        >( char* data, const char& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< int         >( int* data, const int& value );
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< long int    >( long int* data, const long int& value );
#ifdef INSTANTIATE_FLOAT
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< float       >( float* data, const float& value );
#endif
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< double      >( double* data, const double& value );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template void tnlArrayOperations< tnlHost >::setMemoryElement< long double >( long double* data, const long double& value );
#endif

extern template char        tnlArrayOperations< tnlHost >::getMemoryElement< char        >( char* data );
extern template int         tnlArrayOperations< tnlHost >::getMemoryElement< int         >( int* data );
extern template long int    tnlArrayOperations< tnlHost >::getMemoryElement< long int    >( long int* data );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlArrayOperations< tnlHost >::getMemoryElement< float       >( float* data );
#endif
extern template double      tnlArrayOperations< tnlHost >::getMemoryElement< double      >( double* data );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlArrayOperations< tnlHost >::getMemoryElement< long double >( long double* data );
#endif

extern template char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        int >( char* data, const int i );
extern template int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         int >( int* data, const int i );
extern template long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    int >( long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
extern template float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       int >( float* data, const int i );
#endif
extern template double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      int >( double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, int >( long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        long int >( char* data, const long int i );
extern template int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         long int >( int* data, const long int i );
extern template long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    long int >( long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
extern template float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       long int >( float* data, const long int i );
#endif
extern template double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      long int >( double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, long int >( long double* data, const long int i );
#endif
#endif

extern template const char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        int >( const char* data, const int i );
extern template const int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         int >( const int* data, const int i );
extern template const long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    int >( const long int* data, const int i );
#ifdef INSTANTIATE_FLOAT
extern template const float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       int >( const float* data, const int i );
#endif
extern template const double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      int >( const double* data, const int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template const long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, int >( const long double* data, const int i );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template const char&        tnlArrayOperations< tnlHost >::getArrayElementReference< char,        long int >( const char* data, const long int i );
extern template const int&         tnlArrayOperations< tnlHost >::getArrayElementReference< int,         long int >( const int* data, const long int i );
extern template const long int&    tnlArrayOperations< tnlHost >::getArrayElementReference< long int,    long int >( const long int* data, const long int i );
#ifdef INSTANTIATE_FLOAT
extern template const float&       tnlArrayOperations< tnlHost >::getArrayElementReference< float,       long int >( const float* data, const long int i );
#endif
extern template const double&      tnlArrayOperations< tnlHost >::getArrayElementReference< double,      long int >( const double* data, const long int i );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template const long double& tnlArrayOperations< tnlHost >::getArrayElementReference< long double, long int >( const long double* data, const long int i );
#endif
#endif

extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,                char, int >( char* destination, const char* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,                  int, int >( int* destination, const int* source, const int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,        long int, int >( long int* destination, const long int* source, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,              float, int >( float* destination, const float* source, const int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,            double, int >( double* destination, const double* source, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double,  long double, int >( long double* destination, const long double* source, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool tnlArrayOperations< tnlHost >::copyMemory< char,                char, long int >( char* destination, const char* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< int,                  int, long int >( int* destination, const int* source, const long int size );
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long int,        long int, long int >( long int* destination, const long int* source, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::copyMemory< float,              float, long int >( float* destination, const float* source, const long int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::copyMemory< double,            double, long int >( double* destination, const double* source, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::copyMemory< long double,  long double, long int >( long double* destination, const long double* source, const long int size );
#endif
#endif

extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,               char, int >( const char* data1, const char* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,                 int, int >( const int* data1, const int* data2, const int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,       long int, int >( const long int* data1, const long int* data2, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,             float, int >( const float* data1, const float* data2, const int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,           double, int >( const double* data1, const double* data2, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, long double, int >( const long double* data1, const long double* data2, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool tnlArrayOperations< tnlHost >::compareMemory< char,               char, long int >( const char* data1, const char* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< int,                 int, long int >( const int* data1, const int* data2, const long int size );
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long int,       long int, long int >( const long int* data1, const long int* data2, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::compareMemory< float,             float, long int >( const float* data1, const float* data2, const long int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::compareMemory< double,           double, long int >( const double* data1, const double* data2, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::compareMemory< long double, long double, long int >( const long double* data1, const long double* data2, const long int size );
#endif
#endif

extern template bool tnlArrayOperations< tnlHost >::setMemory< char,        int >( char* destination, const char& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< int,         int >( int* destination, const int& value, const int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long int,    int >( long int* destination, const long int& value, const int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::setMemory< float,       int >( float* destination, const float& value, const int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::setMemory< double,      int >( double* destination, const double& value, const int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::setMemory< long double, int >( long double* destination, const long double& value, const int size );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool tnlArrayOperations< tnlHost >::setMemory< char,        long int >( char* destination, const char& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< int,         long int >( int* destination, const int& value, const long int size );
extern template bool tnlArrayOperations< tnlHost >::setMemory< long int,    long int >( long int* destination, const long int& value, const long int size );
#ifdef INSTANTIATE_FLOAT
extern template bool tnlArrayOperations< tnlHost >::setMemory< float,       long int >( float* destination, const float& value, const long int size );
#endif
extern template bool tnlArrayOperations< tnlHost >::setMemory< double,      long int >( double* destination, const double& value, const long int size );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool tnlArrayOperations< tnlHost >::setMemory< long double, long int >( long double* destination, const long double& value, const long int size );
#endif
#endif

#endif


#endif /* TNLARRAYOPERATIONSHOST_IMPL_H_ */
