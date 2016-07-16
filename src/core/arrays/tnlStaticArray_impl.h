/***************************************************************************
                          tnlStaticArray_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSTATICARRAY_IMPL_H_
#define TNLSTATICARRAY_IMPL_H_

#include <core/param-types.h>

template< int Size, typename Element >
__cuda_callable__
inline tnlStaticArray< Size, Element >::tnlStaticArray()
{
};

template< int Size, typename Element >
__cuda_callable__
inline tnlStaticArray< Size, Element >::tnlStaticArray( const Element v[ Size ] )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = v[ i ];
}

template< int Size, typename Element >
__cuda_callable__
inline tnlStaticArray< Size, Element >::tnlStaticArray( const Element& v )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = v;
}

template< int Size, typename Element >
__cuda_callable__
inline tnlStaticArray< Size, Element >::tnlStaticArray( const tnlStaticArray< Size, Element >& v )
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
          ::getType< Element >() +
          tnlString( " >" );
}

template< int Size, typename Element >
__cuda_callable__
inline int tnlStaticArray< Size, Element >::getSize() const
{
   return size;
}

template< int Size, typename Element >
__cuda_callable__
inline Element* tnlStaticArray< Size, Element >::getData()
{
   return data;
}

template< int Size, typename Element >
__cuda_callable__
inline const Element* tnlStaticArray< Size, Element >::getData() const
{
   return data;
}

template< int Size, typename Element >
__cuda_callable__
inline const Element& tnlStaticArray< Size, Element >::operator[]( int i ) const
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< int Size, typename Element >
__cuda_callable__
inline Element& tnlStaticArray< Size, Element >::operator[]( int i )
{
   tnlAssert( i >= 0 && i < size,
            cerr << "i = " << i << " size = " << size << endl; );
   return data[ i ];
}

template< int Size, typename Element >
__cuda_callable__
inline tnlStaticArray< Size, Element >& tnlStaticArray< Size, Element >::operator = ( const tnlStaticArray< Size, Element >& array )
{
   for( int i = 0; i < size; i++ )
      data[ i ] = array[ i ];
   return *this;
}

template< int Size, typename Element >
   template< typename Array >
__cuda_callable__
inline tnlStaticArray< Size, Element >& tnlStaticArray< Size, Element >::operator = ( const Array& array )
{
   for( int i = 0; i < size; i++ )
      data[ i ] = array[ i ];
   return *this;
}

template< int Size, typename Element >
   template< typename Array >
__cuda_callable__
inline bool tnlStaticArray< Size, Element >::operator == ( const Array& array ) const
{
   if( ( int ) size != ( int ) Array::size )
      return false;
   for( int i = 0; i < size; i++ )
      if( data[ i ] != array[ i ] )
         return false;
   return true;
}

template< int Size, typename Element >
   template< typename Array >
__cuda_callable__
inline bool tnlStaticArray< Size, Element >::operator != ( const Array& array ) const
{
   return ! this->operator == ( array );
}

template< int Size, typename Element >
   template< typename OtherElement >
__cuda_callable__
tnlStaticArray< Size, Element >::
operator tnlStaticArray< Size, OtherElement >() const
{
   tnlStaticArray< Size, OtherElement > aux;
   for( int i = 0; i < Size; i++ )
      aux[ i ] = data[ i ];
   return aux;
}

template< int Size, typename Element >
__cuda_callable__
inline void tnlStaticArray< Size, Element >::setValue( const ElementType& val )
{
   for( int i = 0; i < Size; i++ )
      data[ i ] = val;
}

template< int Size, typename Element >
bool tnlStaticArray< Size, Element >::save( tnlFile& file ) const
{
   if( ! file. write< Element, tnlHost, int >( data, size ) )
   {
      cerr << "Unable to write " << getType() << "." << endl;
      return false;
   }
   return true;
}

template< int Size, typename Element >
bool tnlStaticArray< Size, Element >::load( tnlFile& file)
{
   if( ! file.read< Element, tnlHost, int >( data, size ) )
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
ostream& tnlStaticArray< Size, Element >::write( ostream& str, const char* separator ) const
{
   for( int i = 0; i < Size - 1; i++ )
      str << data[ i ] << separator;
   str << data[ Size - 1 ];
   return str;
}


template< int Size, typename Element >
ostream& operator << ( ostream& str, const tnlStaticArray< Size, Element >& a )
{
   a.write( str, "," );
   /*for( int i = 0; i < Size - 1; i ++ )
   {
      str << a[ i ] << ", ";
   }
   str << a[ Size - 1 ];*/
   return str;
};

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: it does not work with CUDA

#ifndef HAVE_CUDA
extern template class tnlStaticArray< 4, char >;
extern template class tnlStaticArray< 4, int >;
#ifdef INSTANTIATE_LONG_INT
extern template class tnlStaticArray< 4, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
extern template class tnlStaticArray< 4, float >;
#endif
extern template class tnlStaticArray< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlStaticArray< 4, long double >;
#endif
#endif

#endif

#endif /* TNLSTATICARRAY_IMPL_H_ */
