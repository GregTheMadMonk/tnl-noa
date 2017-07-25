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

template< typename Element >
__cuda_callable__
inline StaticArray< 1, Element >::StaticArray()
{
}

template< typename Element >
__cuda_callable__
inline StaticArray< 1, Element >::StaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
__cuda_callable__
inline StaticArray< 1, Element >::StaticArray( const Element& v )
{
   data[ 0 ] = v;
}

template< typename Element >
__cuda_callable__
inline StaticArray< 1, Element >::StaticArray( const StaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
String StaticArray< 1, Element >::getType()
{
   return String( "Containers::StaticArray< " ) +
          String( size ) +
          String( ", " ) +
          TNL::getType< Element >() +
          String( " >" );
}

template< typename Element >
__cuda_callable__
inline int StaticArray< 1, Element >::getSize() const
{
   return size;
}

template< typename Element >
__cuda_callable__
inline Element* StaticArray< 1, Element >::getData()
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element* StaticArray< 1, Element >::getData() const
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 1, Element >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 1, Element >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, size, "Element index is out of bounds." );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 1, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 1, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline StaticArray< 1, Element >& StaticArray< 1, Element >::operator = ( const StaticArray< 1, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline StaticArray< 1, Element >& StaticArray< 1, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 1, Element >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size && data[ 0 ] == array[ 0 ] );
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool StaticArray< 1, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
   template< typename OtherElement >
__cuda_callable__
StaticArray< 1, Element >::
operator StaticArray< 1, OtherElement >() const
{
   StaticArray< 1, OtherElement > aux;
   aux[ 0 ] = data[ 0 ];
   return aux;
}


template< typename Element >
__cuda_callable__
inline void StaticArray< 1, Element >::setValue( const ElementType& val )
{
   data[ 0 ] = val;
}

template< typename Element >
bool StaticArray< 1, Element >::save( File& file ) const
{
   if( ! file. write< Element, Devices::Host, int >( data, size ) )
   {
      std::cerr << "Unable to write " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
bool StaticArray< 1, Element >::load( File& file)
{
   if( ! file.read( data, size ) )
   {
      std::cerr << "Unable to read " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
void StaticArray< 1, Element >::sort()
{
}

template< typename Element >
std::ostream& StaticArray< 1, Element >::write( std::ostream& str, const char* separator ) const
{
   str << data[ 0 ];
   return str;
}

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: it does not work with CUDA

#ifndef HAVE_CUDA
extern template class StaticArray< 1, char >;
extern template class StaticArray< 1, int >;
#ifdef INSTANTIATE_LONG_INT
extern template class StaticArray< 1, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
extern template class StaticArray< 1, float >;
#endif
extern template class StaticArray< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticArray< 1, long double >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
