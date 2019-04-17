/***************************************************************************
                          StaticArray3D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/param-types.h>
#include <TNL/Math.h>
#include <TNL/Containers/StaticArray.h>

namespace TNL {
namespace Containers {

template< typename Value >
__cuda_callable__
constexpr int StaticArray< 3, Value >::getSize()
{
   return size;
}

template< typename Value >
__cuda_callable__
inline StaticArray< 3, Value >::StaticArray()
{
}

template< typename Value >
   template< typename _unused >
__cuda_callable__
inline StaticArray< 3, Value >::StaticArray( const Value v[ size ] )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Value >
__cuda_callable__
inline StaticArray< 3, Value >::StaticArray( const Value& v )
{
   data[ 0 ] = v;
   data[ 1 ] = v;
   data[ 2 ] = v;
}

template< typename Value >
__cuda_callable__
inline StaticArray< 3, Value >::StaticArray( const Value& v1, const Value& v2, const Value& v3 )
{
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< typename Value >
__cuda_callable__
inline StaticArray< 3, Value >::StaticArray( const StaticArray< size, Value >& v )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Value >
String StaticArray< 3, Value >::getType()
{
   return String( "Containers::StaticArray< " ) +
          convertToString( size ) +
          String( ", " ) +
          TNL::getType< Value >() +
          String( " >" );
}

template< typename Value >
__cuda_callable__
inline Value* StaticArray< 3, Value >::getData()
{
   return data;
}

template< typename Value >
__cuda_callable__
inline const Value* StaticArray< 3, Value >::getData() const
{
   return data;
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 3, Value >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 3, Value >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 3, Value >::x()
{
   return data[ 0 ];
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 3, Value >::x() const
{
   return data[ 0 ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 3, Value >::y()
{
   return data[ 1 ];
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 3, Value >::y() const
{
   return data[ 1 ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 3, Value >::z()
{
   return data[ 2 ];
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 3, Value >::z() const
{
   return data[ 2 ];
}
template< typename Value >
__cuda_callable__
StaticArray< 3, Value >& StaticArray< 3, Value >::operator = ( const StaticArray< 3, Value >& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
   return *this;
}

template< typename Value >
   template< typename Array >
__cuda_callable__
StaticArray< 3, Value >& StaticArray< 3, Value >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
   return *this;
}

template< typename Value >
   template< typename Array >
__cuda_callable__
bool StaticArray< 3, Value >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size &&
           data[ 0 ] == array[ 0 ] &&
           data[ 1 ] == array[ 1 ] &&
           data[ 2 ] == array[ 2 ] );
}

template< typename Value >
   template< typename Array >
__cuda_callable__
bool StaticArray< 3, Value >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Value >
   template< typename OtherValue >
__cuda_callable__
StaticArray< 3, Value >::
operator StaticArray< 3, OtherValue >() const
{
   StaticArray< 3, OtherValue > aux;
   aux[ 0 ] = data[ 0 ];
   aux[ 1 ] = data[ 1 ];
   aux[ 2 ] = data[ 2 ];
   return aux;
}

template< typename Value >
__cuda_callable__
void StaticArray< 3, Value >::setValue( const ValueType& val )
{
   data[ 2 ] = data[ 1 ] = data[ 0 ] = val;
}

template< typename Value >
bool StaticArray< 3, Value >::save( File& file ) const
{
   file.save< Value, Value, Devices::Host >( data, size );
   return true;
}

template< typename Value >
bool StaticArray< 3, Value >::load( File& file)
{
   file.load< Value, Value, Devices::Host >( data, size );
   return true;
}

template< typename Value >
void StaticArray< 3, Value >::sort()
{
   /****
    * Bubble sort on three elements
    */
   if( data[ 0 ] > data[ 1 ] )
      swap( data[ 0 ], data[ 1 ] );
   if( data[ 1 ] > data[ 2 ] )
      swap( data[ 1 ], data[2  ] );
   if( data[ 0 ] > data[ 1 ] )
      swap( data[ 0 ], data[ 1 ] );
}

template< typename Value >
std::ostream& StaticArray< 3, Value >::write( std::ostream& str, const char* separator ) const
{
   str << data[ 0 ] << separator << data[ 1 ] << separator << data[ 2 ];
   return str;
}

} // namespace Containers
} // namespace TNL
