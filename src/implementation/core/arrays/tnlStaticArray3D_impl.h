/***************************************************************************
                          tnlStaticArray3D_impl.h  -  description
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

#ifndef TNLSTATICARRAY3D_IMPL_H_
#define TNLSTATICARRAY3D_IMPL_H_

#include <core/param-types.h>

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 3, Element >::tnlStaticArray()
{
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 3, Element >::tnlStaticArray( const Element v[ size ] )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 3, Element >::tnlStaticArray( const Element& v )
{
   data[ 0 ] = v;
   data[ 1 ] = v;
   data[ 2 ] = v;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 3, Element >::tnlStaticArray( const Element& v1, const Element& v2, const Element& v3 )
{
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< 3, Element >::tnlStaticArray( const tnlStaticArray< size, Element >& v )
{
   data[ 0 ] = v[ 0 ];
   data[ 1 ] = v[ 1 ];
   data[ 2 ] = v[ 2 ];
}

template< typename Element >
tnlString tnlStaticArray< 3, Element >::getType()
{
   return tnlString( "tnlStaticArray< " ) +
          tnlString( size ) +
          tnlString( ", " ) +
          ::getType< Element >() +
          tnlString( " >" );
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
int tnlStaticArray< 3, Element >::getSize() const
{
   return size;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
Element* tnlStaticArray< 3, Element >::getData()
{
   return data;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< typename Element >
const Element* tnlStaticArray< 3, Element >::getData() const
{
   return data;
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 3, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 3, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 3, Element >::x()
{
   return data[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 3, Element >::x() const
{
   return data[ 0 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 3, Element >::y()
{
   return data[ 1 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 3, Element >::y() const
{
   return data[ 1 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< 3, Element >::z()
{
   return data[ 2 ];
}

template< typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< 3, Element >::z() const
{
   return data[ 2 ];
}
template< typename Element >
tnlStaticArray< 3, Element >& tnlStaticArray< 3, Element >::operator = ( const tnlStaticArray< 3, Element >& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
}

template< typename Element >
   template< typename Array >
tnlStaticArray< 3, Element >& tnlStaticArray< 3, Element >::operator = ( const Array& array )
{
   data[ 0 ] = array[ 0 ];
   data[ 1 ] = array[ 1 ];
   data[ 2 ] = array[ 2 ];
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 3, Element >::operator == ( const Array& array ) const
{
   return( size == Array::size &&
           data[ 0 ] == array[ 0 ] &&
           data[ 1 ] == array[ 1 ] &&
           data[ 2 ] == array[ 2 ] );
}

template< typename Element >
   template< typename Array >
bool tnlStaticArray< 3, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< typename Element >
void tnlStaticArray< 3, Element >::setValue( const ElementType& val )
{
   data[ 2 ] = data[ 1 ] = data[ 0 ] = val;
}

template< typename Element >
bool tnlStaticArray< 3, Element >::save( tnlFile& file ) const
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
bool tnlStaticArray< 3, Element >::load( tnlFile& file)
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
void tnlStaticArray< 3, Element >::sort()
{
   /****
    * Bubble sort on three elements
    */
   if( data[ 0 ] > data[ 1 ] )
      Swap( data[ 0 ], data[ 1 ] );
   if( data[ 1 ] > data[ 2 ] )
      Swap( data[ 1 ], data[2  ] );
   if( data[ 0 ] > data[ 1 ] )
      Swap( data[ 0 ], data[ 1 ] );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticArray< 3, char >;
extern template class tnlStaticArray< 3, int >;
extern template class tnlStaticArray< 3, long int >;
extern template class tnlStaticArray< 3, float >;
extern template class tnlStaticArray< 3, double >;
extern template class tnlStaticArray< 3, long double >;

#endif


#endif /* TNLSTATICARRAY3D_IMPL_H_ */
