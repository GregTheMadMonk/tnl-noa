/***************************************************************************
                          tnlStaticArray2D_impl.h  -  description
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

#ifndef TNLSTATICARRAY2D_IMPL_H_
#define TNLSTATICARRAY2D_IMPL_H_

#include <core/param-types.h>

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 2, Element >::tnlStaticArray()
{
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 2, Element >::tnlStaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 2, Element >::tnlStaticArray( const Element& v )
{
   data[ 0 ] = v;
   data[ 1 ] = v;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 2, Element >::tnlStaticArray( const Element& v1, const Element& v2 )
{
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 2, Element >::tnlStaticArray( const tnlStaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
}

template< typename Element >
tnlString tnlStaticArray< 2, Element >::getType()
{
   return tnlString( "tnlStaticArray< " ) +
          tnlString( size ) +
          tnlString( ", " ) +
          getParameterType< Element >() +
          tnlString( " >" );
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 2, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 2, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 2, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 2, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 2, Element >::y()
{
   return data[ 1 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 2, Element >::y() const
{
   return data[ 1 ];
}

template< typename Element >
tnlStaticArray< 2, Element >& tnlStaticArray< 2, Element >::operator = ( const tnlStaticArray< 2, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
}

template< typename Element >
   template< typename Array >
tnlStaticArray< 2, Element >& tnlStaticArray< 2, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 2, Element >::operator == ( const Array& array ) const
{
   return( size == Array::size &&
           data[ 0 ] == array[ 0 ] &&
           data[ 1 ] == array[ 1 ] );
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 2, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
bool tnlStaticArray< 2, Element >::save( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   if( ! file. write< Element, tnlHost, int >( data, size ) )
      cerr << "Unable to write " << getType() << "." << endl;
#else
   if( ! file. write( data, size ) )
      cerr << "Unable to write " << getType() << "." << endl;
#endif
   return true;
}

template< typename Element >
bool tnlStaticArray< 2, Element >::load( tnlFile& file)
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

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticArray< 2, char >;
extern template class tnlStaticArray< 2, int >;
extern template class tnlStaticArray< 2, long int >;
extern template class tnlStaticArray< 2, float >;
extern template class tnlStaticArray< 2, double >;
extern template class tnlStaticArray< 2, long double >;

#endif


#endif /* TNLSTATICARRAY2D_IMPL_H_ */
