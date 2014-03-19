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
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 1, Element >::tnlStaticArray()
{
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 1, Element >::tnlStaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 1, Element >::tnlStaticArray( const Element& v )
{
   data[ 0 ] = v;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 1, Element >::tnlStaticArray( const tnlStaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
}

template< typename Element >
tnlString tnlStaticArray< 1, Element >::getType()
{
   return tnlString( "tnlStaticArray< " ) +
          tnlString( size ) +
          tnlString( ", " ) +
          getParameterType< Element >() +
          tnlString( " >" );
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
int tnlStaticArray< 1, Element >::getSize() const
{
   return size;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
Element* tnlStaticArray< 1, Element >::getData()
{
   return data;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
const Element* tnlStaticArray< 1, Element >::getData() const
{
   return data;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 1, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 1, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 1, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 1, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
tnlStaticArray< 1, Element >& tnlStaticArray< 1, Element >::operator = ( const tnlStaticArray< 1, Element >& array )
{
   data[ 0 ] = array[ 0 ];
}

template< typename Element >
   template< typename Array >
tnlStaticArray< 1, Element >& tnlStaticArray< 1, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 1, Element >::operator == ( const Array& array ) const
{
   return( size == Array::size && data[ 0 ] == array[ 0 ] );
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 1, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
void tnlStaticArray< 1, Element >::setValue( const ElementType& val )
{
   data[ 0 ] = val;
}

template< typename Element >
bool tnlStaticArray< 1, Element >::save( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   if( ! file. write< Element, tnlHost, int >( data, size ) )
#else
   if( ! file. write( data, size ) )
#endif
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

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticArray< 1, char >;
extern template class tnlStaticArray< 1, int >;
extern template class tnlStaticArray< 1, long int >;
extern template class tnlStaticArray< 1, float >;
extern template class tnlStaticArray< 1, double >;
extern template class tnlStaticArray< 1, long double >;

#endif

#endif /* TNLSTATICARRAY1D_IMPL_H_ */
