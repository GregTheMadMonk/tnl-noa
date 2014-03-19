/***************************************************************************
                          tnlStaticArray_impl.h  -  description
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

#ifndef TNLSTATICARRAY_IMPL_H_
#define TNLSTATICARRAY_IMPL_H_

#include <core/param-types.h>

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< Size, Element >::tnlStaticArray()
{
};

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< Size, Element >::tnlStaticArray( const Element v[ Size ] )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = v[ i ];
}

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< Size, Element >::tnlStaticArray( const Element& v )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = v;
}

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticArray< Size, Element >::tnlStaticArray( const tnlStaticArray< Size, Element >& v )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = v[ i ];
}

template< int Size, typename Element >
tnlString tnlStaticArray< Size, Element >::getType()
{
   return tnlString( "tnlStaticArray< " ) +
          tnlString( Size ) +
          tnlString( ", " ) +
          getParameterType< Element >() +
          tnlString( " >" );
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< int Size, typename Element >
int tnlStaticArray< Size, Element >::getSize() const
{
   return size;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< int Size, typename Element >
Element* tnlStaticArray< Size, Element >::getData()
{
   return data;
}

#ifdef HAVE_CUDA
   __host__ __device__
#endif
template< int Size, typename Element >
const Element* tnlStaticArray< Size, Element >::getData() const
{
   return data;
}

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
const Element& tnlStaticArray< Size, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< int Size, typename Element >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Element& tnlStaticArray< Size, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< int Size, typename Element >
tnlStaticArray< Size, Element >& tnlStaticArray< Size, Element >::operator = ( const tnlStaticArray< Size, Element >& array )
{
   for( int i = 0; i < size; i++ )
      data[ i ] = array[ i ];
}

template< int Size, typename Element >
   template< typename Array >
tnlStaticArray< Size, Element >& tnlStaticArray< Size, Element >::operator = ( const Array& array )
{
   for( int i = 0; i < size; i++ )
      data[ i ] = array[ i ];
}

template< int Size, typename Element >
   template< typename Array >
bool tnlStaticArray< Size, Element >::operator == ( const Array& array ) const
{
   if( size != Array::size )
      return false;
   for( int i = 0; i < size; i++ )
      if( data[ i ] != array[ i ] )
         return false;
   return true;
}

template< int Size, typename Element >
   template< typename Array >
bool tnlStaticArray< Size, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< int Size, typename Element >
void tnlStaticArray< Size, Element >::setValue( const ElementType& val )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = val;
}

template< int Size, typename Element >
bool tnlStaticArray< Size, Element >::save( tnlFile& file ) const
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

template< int Size, typename Element >
bool tnlStaticArray< Size, Element >::load( tnlFile& file)
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

template< int Size, typename Element >
void tnlStaticArray< Size, Element >::sort()
{
   /****
    * We assume that the array data is small and so
    * may sort it with the bubble sort.
    */
   for( int k = Size - 1; k > 0; k--)
      for( int i = 0; i < k; i++ )
         if( data[ i ] > data[ i+1 ] )
            Swap( data[ i ], data[ i+1 ] );
}


template< int Size, typename Element >
ostream& operator << ( ostream& str, const tnlStaticArray< Size, Element >& a )
{
   for( int i = 0; i < Size - 1; i ++ )
   {
      str << a[ i ] << ", ";
   }
   str << a[ Size - 1 ];
   return str;
};

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticArray< 4, char >;
extern template class tnlStaticArray< 4, int >;
extern template class tnlStaticArray< 4, long int >;
extern template class tnlStaticArray< 4, float >;
extern template class tnlStaticArray< 4, double >;
extern template class tnlStaticArray< 4, long double >;

#endif

#endif /* TNLSTATICARRAY_IMPL_H_ */
