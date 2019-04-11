/***************************************************************************
                          StaticArray1D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/param-types.h>
#include <TNL/Containers/StaticArray.h>

namespace TNL {
namespace Containers {   

template< typename Value >
__cuda_callable__
inline StaticArray< 1, Value >::StaticArray()
{
}

template< typename Value >
   template< typename _unused >
__cuda_callable__
inline StaticArray< 1, Value >::StaticArray( const Value v[ size ] )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Value >
__cuda_callable__
inline StaticArray< 1, Value >::StaticArray( const Value& v )
{
   data[ 0 ] = v;
}

template< typename Value >
__cuda_callable__
inline StaticArray< 1, Value >::StaticArray( const StaticArray< size, Value >& v )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Value >
String StaticArray< 1, Value >::getType()
{
   return String( "Containers::StaticArray< " ) +
          convertToString( size ) +
          String( ", " ) +
          TNL::getType< Value >() +
          String( " >" );
}

template< typename Value >
__cuda_callable__
inline int StaticArray< 1, Value >::getSize() const
{
   return size;
}

template< typename Value >
__cuda_callable__
inline Value* StaticArray< 1, Value >::getData()
{
   return data;
}

template< typename Value >
__cuda_callable__
inline const Value* StaticArray< 1, Value >::getData() const
{
   return data;
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 1, Value >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 1, Value >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Value >
__cuda_callable__
inline Value& StaticArray< 1, Value >::x()
{
   return data[ 0 ];
}

template< typename Value >
__cuda_callable__
inline const Value& StaticArray< 1, Value >::x() const
{
   return data[ 0 ];
}

template< typename Value >
__cuda_callable__
inline StaticArray< 1, Value >& StaticArray< 1, Value >::operator = ( const StaticArray< 1, Value >& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Value >
   template< typename Array >
__cuda_callable__
inline StaticArray< 1, Value >& StaticArray< 1, Value >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Value >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 1, Value >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size && data[ 0 ] == array[ 0 ] );
}

template< typename Value >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 1, Value >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Value >
   template< typename OtherValue >
__cuda_callable__
StaticArray< 1, Value >::
operator StaticArray< 1, OtherValue >() const
{
   StaticArray< 1, OtherValue > aux;
   aux[ 0 ] = data[ 0 ];
   return aux;
}


template< typename Value >
__cuda_callable__
inline void StaticArray< 1, Value >::setValue( const ValueType& val )
{
   data[ 0 ] = val;
}

template< typename Value >
bool StaticArray< 1, Value >::save( File& file ) const
{
   file.save< Value, Value, Devices::Host >( data, size );
   return true;
}

template< typename Value >
bool StaticArray< 1, Value >::load( File& file)
{
   file.load< Value, Value, Devices::Host >( data, size );
   return true;
}

template< typename Value >
void StaticArray< 1, Value >::sort()
{
}

template< typename Value >
std::ostream& StaticArray< 1, Value >::write( std::ostream& str, const char* separator ) const
{
   str << data[ 0 ];
   return str;
}

} // namespace Containers
} // namespace TNL
