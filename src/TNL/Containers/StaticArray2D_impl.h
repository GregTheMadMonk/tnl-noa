/***************************************************************************
                          StaticArray2D_impl.h  -  description
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

template< typename Element >
__cuda_callable__
inline StaticArray< 2, Element >::StaticArray()
{
}

template< typename Element >
   template< typename _unused >
__cuda_callable__
inline StaticArray< 2, Element >::StaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
}

template< typename Element >
__cuda_callable__
inline StaticArray< 2, Element >::StaticArray( const Element& v )
{
   data[ 0 ] = v;
   data[ 1 ] = v;
}

template< typename Element >
__cuda_callable__
inline StaticArray< 2, Element >::StaticArray( const Element& v1, const Element& v2 )
{
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< typename Element >
__cuda_callable__
inline StaticArray< 2, Element >::StaticArray( const StaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
}

template< typename Element >
String StaticArray< 2, Element >::getType()
{
   return String( "Containers::StaticArray< " ) +
          String( size ) +
          String( ", " ) +
          TNL::getType< Element >() +
          String( " >" );
}

template< typename Element >
__cuda_callable__
inline int StaticArray< 2, Element >::getSize() const
{
   return size;
}

template< typename Element >
__cuda_callable__
inline Element* StaticArray< 2, Element >::getData()
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element* StaticArray< 2, Element >::getData() const
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 2, Element >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 2, Element >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 2, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 2, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 2, Element >::y()
{
   return data[ 1 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 2, Element >::y() const
{
   return data[ 1 ];
}

template< typename Element >
__cuda_callable__
inline StaticArray< 2, Element >& StaticArray< 2, Element >::operator = ( const StaticArray< 2, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline StaticArray< 2, Element >& StaticArray< 2, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 2, Element >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size &&
           data[ 0 ] == array[ 0 ] &&
           data[ 1 ] == array[ 1 ] );
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 2, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
   template< typename OtherElement >
__cuda_callable__
StaticArray< 2, Element >::
operator StaticArray< 2, OtherElement >() const
{
   StaticArray< 2, OtherElement > aux;
   aux[ 0 ] = data[ 0 ];
   aux[ 1 ] = data[ 1 ];
   return aux;
}

template< typename Element >
__cuda_callable__
inline void StaticArray< 2, Element >::setValue( const ElementType& val )
{
   data[ 1 ] = data[ 0 ] = val;
}

template< typename Element >
bool StaticArray< 2, Element >::save( File& file ) const
{
   if( ! file. write< Element, Devices::Host, int >( data, size ) )
   {
      std::cerr << "Unable to write " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
bool StaticArray< 2, Element >::load( File& file)
{
   if( ! file.read< Element, Devices::Host, int >( data, size ) )
   {
      std::cerr << "Unable to read " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
void StaticArray< 2, Element >::sort()
{
   if( data[ 0 ] > data[ 1 ] )
      swap( data[ 0 ], data[ 1 ] );
}

template< typename Element >
std::ostream& StaticArray< 2, Element >::write( std::ostream& str, const char* separator ) const
{
   str << data[ 0 ] << separator << data[ 1 ];
   return str;
}

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: it does not work with CUDA

#ifndef HAVE_CUDA
extern template class StaticArray< 2, char >;
extern template class StaticArray< 2, int >;
#ifdef INSTANTIATE_LONG_INT
extern template class StaticArray< 2, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
extern template class StaticArray< 2, float >;
#endif
extern template class StaticArray< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticArray< 2, long double >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
