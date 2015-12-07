/***************************************************************************
                          tnlStaticArray1D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSTATICARRAY1D_IMPL_H_
#define TNLSTATICARRAY1D_IMPL_H_

#include <core/param-types.h>

template< typename Element >
__cuda_callable__
inline tnlStaticArray< 1, Element >::tnlStaticArray()
{
}

template< typename Element >
__cuda_callable__
inline tnlStaticArray< 1, Element >::tnlStaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
__cuda_callable__
inline tnlStaticArray< 1, Element >::tnlStaticArray( const Element& v )
{
   data[ 0 ] = v;
}

template< typename Element >
__cuda_callable__
inline tnlStaticArray< 1, Element >::tnlStaticArray( const tnlStaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
tnlString tnlStaticArray< 1, Element >::getType()
{
   return tnlString( "tnlStaticArray< " ) +
          tnlString( size ) +
          tnlString( ", " ) +
          ::getType< Element >() +
          tnlString( " >" );
}

template< typename Element >
__cuda_callable__
inline int tnlStaticArray< 1, Element >::getSize() const
{
   return size;
}

template< typename Element >
__cuda_callable__
inline Element* tnlStaticArray< 1, Element >::getData()
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element* tnlStaticArray< 1, Element >::getData() const
{
   return data;
}

template< typename Element >
__cuda_callable__
inline const Element& tnlStaticArray< 1, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& tnlStaticArray< 1, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
__cuda_callable__
inline Element& tnlStaticArray< 1, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline const Element& tnlStaticArray< 1, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
__cuda_callable__
inline tnlStaticArray< 1, Element >& tnlStaticArray< 1, Element >::operator = ( const tnlStaticArray< 1, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline tnlStaticArray< 1, Element >& tnlStaticArray< 1, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   return *this;
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool tnlStaticArray< 1, Element >::operator == ( const Array& array ) const
{
   return( ( int ) size == ( int ) Array::size && data[ 0 ] == array[ 0 ] );
}

template< typename Element >
   template< typename Array >
__cuda_callable__
inline bool tnlStaticArray< 1, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
   template< typename OtherElement >
__cuda_callable__
tnlStaticArray< 1, Element >::
operator tnlStaticArray< 1, OtherElement >() const
{
   tnlStaticArray< 1, OtherElement > aux;
   aux[ 0 ] = data[ 0 ];
   return aux;
}


template< typename Element >
__cuda_callable__
inline void tnlStaticArray< 1, Element >::setValue( const ElementType& val )
{
   data[ 0 ] = val;
}

template< typename Element >
bool tnlStaticArray< 1, Element >::save( tnlFile& file ) const
{
   if( ! file. write< Element, tnlHost, int >( data, size ) )
   {
      cerr << "Unable to write " << getType() << "." << endl;
      return false;
   }
   return true;
}

template< typename Element >
bool tnlStaticArray< 1, Element >::load( tnlFile& file)
{
#ifdef HAVE_NOT_CXX11
   if( ! file.read< Element, tnlHost, int >( data, size ) )
#else
   if( ! file.read( data, size ) )
#endif
   {
      cerr << "Unable to read " << getType() << "." << endl;
      return false;
   }
   return true;
}

template< typename Element >
void tnlStaticArray< 1, Element >::sort()
{
}

template< typename Element >
ostream& tnlStaticArray< 1, Element >::write( ostream& str, const char* separator ) const
{
   str << data[ 0 ];
   return str;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: it does not work with CUDA

#ifndef HAVE_CUDA
extern template class tnlStaticArray< 1, char >;
extern template class tnlStaticArray< 1, int >;
#ifdef INSTANTIATE_LONG_INT
extern template class tnlStaticArray< 1, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
extern template class tnlStaticArray< 1, float >;
#endif
extern template class tnlStaticArray< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlStaticArray< 1, long double >;
#endif
#endif

#endif

#endif /* TNLSTATICARRAY1D_IMPL_H_ */
