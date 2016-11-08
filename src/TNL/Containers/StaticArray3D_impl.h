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

namespace TNL {
namespace Containers {   

template< typename Element >
__cuda_callable__
inline StaticArray< 3, Element >::StaticArray()
{
}

template< typename Element >
__cuda_callable__
inline StaticArray< 3, Element >::StaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Element >
__cuda_callable__
inline StaticArray< 3, Element >::StaticArray( const Element& v )
{
   data[ 0 ] = v;
   data[ 1 ] = v;
   data[ 2 ] = v;
}

template< typename Element >
__cuda_callable__
inline StaticArray< 3, Element >::StaticArray( const Element& v1, const Element& v2, const Element& v3 )
{
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< typename Element >
__cuda_callable__
inline StaticArray< 3, Element >::StaticArray( const StaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Element >
String StaticArray< 3, Element >::getType()
{
   return String( "Containers::StaticArray< " ) +
          String( size ) +
          String( ", " ) +
          TNL::getType< Element >() +
          String( " >" );
}

template< typename Element >
inline int StaticArray< 3, Element >::getSize() const
{
   return size;
}

template< typename Element >
__cuda_callable__
inline Element* StaticArray< 3, Element >::getData()
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element* StaticArray< 3, Element >::getData() const
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 3, Element >::operator[]( int i ) const
{
   Assert( i >= 0 && i < size,
            std::cerr << "i = " << i << " size = " << size << std::endl; );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 3, Element >::operator[]( int i )
{
   Assert( i >= 0 && i < size,
            std::cerr << "i = " << i << " size = " << size << std::endl; );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 3, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 3, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 3, Element >::y()
{
   return data[ 1 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 3, Element >::y() const
{
   return data[ 1 ];
}

template< typename Element >
__cuda_callable__
inline Element& StaticArray< 3, Element >::z()
{
   return data[ 2 ];
}

template< typename Element >
__cuda_callable__
inline const Element& StaticArray< 3, Element >::z() const
{
   return data[ 2 ];
}
template< typename Element >
__cuda_callable__
StaticArray< 3, Element >& StaticArray< 3, Element >::operator = ( const StaticArray< 3, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
StaticArray< 3, Element >& StaticArray< 3, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
bool StaticArray< 3, Element >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size &&
           data[ 0 ] == array[ 0 ] &&
           data[ 1 ] == array[ 1 ] &&
           data[ 2 ] == array[ 2 ] );
}

template< typename Element >
   template< typename Array >
__cuda_callable__
bool StaticArray< 3, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
   template< typename OtherElement >
__cuda_callable__
StaticArray< 3, Element >::
operator StaticArray< 3, OtherElement >() const
{
   StaticArray< 3, OtherElement > aux;
   aux[ 0 ] = data[ 0 ];
   aux[ 1 ] = data[ 1 ];
   aux[ 2 ] = data[ 2 ];
   return aux;
}

template< typename Element >
__cuda_callable__
void StaticArray< 3, Element >::setValue( const ElementType& val )
{
   data[ 2 ] = data[ 1 ] = data[ 0 ] = val;
}

template< typename Element >
bool StaticArray< 3, Element >::save( File& file ) const
{
   if( ! file. write< Element, Devices::Host, int >( data, size ) )
   {
      std::cerr << "Unable to write " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
bool StaticArray< 3, Element >::load( File& file)
{
   if( ! file.read< Element, Devices::Host, int >( data, size ) )
   {
      std::cerr << "Unable to read " << getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Element >
void StaticArray< 3, Element >::sort()
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

template< typename Element >
std::ostream& StaticArray< 3, Element >::write( std::ostream& str, const char* separator ) const
{
   str << data[ 0 ] << separator << data[ 1 ] << separator << data[ 2 ];
   return str;
}


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: it does not work with CUDA

#ifndef HAVE_CUDA
extern template class StaticArray< 3, char >;
extern template class StaticArray< 3, int >;
#ifdef INSTANTIATE_LONG_INT
extern template class StaticArray< 3, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
extern template class StaticArray< 3, float >;
#endif
extern template class StaticArray< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticArray< 3, long double >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
