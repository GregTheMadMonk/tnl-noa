/***************************************************************************
                          tnlList_impl.h  -  description
                             -------------------
    begin                : Mar, 5 Apr 2016 12:46 PM
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlFile.h>

namespace TNL {

template< typename T >
tnlList< T >::tnlList()
   : first( 0 ),  last( 0 ), size( 0 ), iterator( 0 ), index( 0 )
{
}

template< typename T >
tnlList< T >::tnlList( const tnlList& list )
   : first( 0 ), last( 0 ), size( 0 ), iterator( 0 ), index( 0 )
{
   AppendList( list );
}

template< typename T >
tnlList< T >::~tnlList()
{
   reset();
}

template< typename T >
tnlString tnlList< T >::getType()
{
   return tnlString( "tnlList< " ) + TNL::getType< T >() +  tnlString( " >" );
}

template< typename T >
bool tnlList< T >::isEmpty() const
{
   return ! size;
}
 
template< typename T >
int tnlList< T >::getSize() const
{
   return size;
}

template< typename T >
T& tnlList< T >::operator[]( const int& ind )
{
   tnlAssert( ind < size, );
   int iter_dist = abs( index - ind );
   if( ! iterator ||
       iter_dist > ind ||
       iter_dist > size - ind )
   {
      if( ind < size - ind )
      {
         //cout << "Setting curent index to 0." << std::endl;
         index = 0;
         iterator = first;
      }
      else
      {
         //cout << "Setting curent index to size - 1." << std::endl;
         index = size - 1;
         iterator = last;
      }
   }
   while( index != ind )
   {
      //cout << " current index = " << index
      //     << " index = " << ind << std::endl;
      if( ind < index )
      {
         iterator = iterator -> Previous();
         index --;
      }
      else
      {
         iterator = iterator -> Next();
         index ++;
      }
      tnlAssert( iterator, );
   }
   return iterator -> Data();
};
 
template< typename T >
const T& tnlList< T >::operator[]( const int& ind ) const
{
   return const_cast< tnlList< T >* >( this ) -> operator[]( ind );
}

template< typename T >
const tnlList< T >& tnlList< T >::operator = ( const tnlList& lst )
{
   AppendList( lst );
   return( *this );
}

template< typename T >
bool tnlList< T >::Append( const T& data )
{
   if( ! first )
   {
      tnlAssert( ! last, );
      first = last = new tnlDataElement< T >( data );
      if( ! first ) return false;
   }
   else
   {
      tnlDataElement< T >* new_element =  new tnlDataElement< T >( data, last, 0 );
      if( ! new_element ) return false;
      tnlAssert( last, );
      last = last -> Next() = new_element;
   }
   size ++;
   return true;
};

template< typename T >
bool tnlList< T >::Prepend( const T& data )
{
   if( ! first )
   {
      tnlAssert( ! last, );
      first = last = new tnlDataElement< T >( data );
      if( ! first ) return false;
   }
   else
   {
      tnlDataElement< T >* new_element =  new tnlDataElement< T >( data, 0, first );
      if( ! new_element ) return false;
      first = first -> Previous() = new_element;
   }
   size ++;
   index ++;
   return true;
};

template< typename T >
bool tnlList< T >::Insert( const T& data, const int& ind )
{
   tnlAssert( ind <= size || ! size, );
   if( ind == 0 ) return Prepend( data );
   if( ind == size ) return Append( data );
   operator[]( ind );
   tnlDataElement< T >* new_el =
      new tnlDataElement< T >( data,
                             iterator -> Previous(),
                             iterator );
   if( ! new_el ) return false;
   iterator -> Previous() -> Next() = new_el;
   iterator -> Previous() = new_el;
   iterator = new_el;
   size ++;
   return true;
};

template< typename T >
bool tnlList< T >::AppendList( const tnlList< T >& lst )
{
   int i;
   for( i = 0; i < lst. getSize(); i ++ )
   {
      if( ! Append( lst[ i ] ) ) return false;
   }
   return true;
};
 
template< typename T >
bool tnlList< T >::PrependList( const tnlList< T >& lst )

{
   int i;
   for( i = lst. getSize(); i > 0; i -- )
      if( ! Prepend( lst[ i - 1 ] ) ) return false;
   return true;
};

template< typename T >
   template< typename Array >
void tnlList< T >::toArray( Array& array )
{
   tnlAssert( this->getSize() <= array.getSize(),
              std::cerr << "this->getSize() = " << this->getSize()
                   << " array.getSize() = " << array.getSize() << std::endl; );
   for( int i = 0; i < this->getSize(); i++ )
      array[ i ] = ( *this )[ i ];
}

template< typename T >
void tnlList< T >::Erase( const int& ind )
{
   operator[]( ind );
   tnlDataElement< T >* tmp_it = iterator;
   if( iterator -> Next() )
      iterator -> Next() -> Previous() = iterator -> Previous();
   if( iterator -> Previous() )
     iterator -> Previous() -> Next() = iterator -> Next();
   if( iterator -> Next() ) iterator = iterator -> Next();
   else
   {
      iterator = iterator -> Previous();
      index --;
   }
   if( first == tmp_it ) first = iterator;
   if( last == tmp_it ) last = iterator;
   delete tmp_it;
   size --;
};

template< typename T >
void tnlList< T >::DeepErase( const int& ind )
{
   operator[]( ind );
   delete iterator -> Data();
   Erase( ind );
};

template< typename T >
void tnlList< T >::reset()
{
   iterator = first;
   tnlDataElement< T >* tmp_it;
   while( iterator )
   {
      tnlAssert( iterator, );
      tmp_it = iterator;
      iterator = iterator -> Next();
      delete tmp_it;
   }
   first = last = 0;
   size = 0;
};

template< typename T >
void tnlList< T >::DeepEraseAll()
{
   iterator = first;
   tnlDataElement< T >* tmp_it;
   while( iterator )
   {
      tmp_it = iterator;
      iterator = iterator -> Next();
      delete tmp_it -> Data();
      delete tmp_it;
   }
   first = last = 0;
   size = 0;
};
 
template< typename T >
bool tnlList< T >::Save( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   file. write< const int, tnlHost >( &size );
   for( int i = 0; i < size; i ++ )
      if( ! file. write< int, tnlHost, int >( &operator[]( i ), 1 ) )
         return false;
   return true;
#else
   file. write( &size );
   for( int i = 0; i < size; i ++ )
      if( ! file. write( &operator[]( i ), 1 ) )
         return false;
   return true;

#endif
}

template< typename T >
bool tnlList< T >::DeepSave( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   file. write< const int, tnlHost >( &size );
   for( int i = 0; i < size; i ++ )
      if( ! operator[]( i ). save( file ) ) return false;
   return true;
#else
   file. write( &size );
   for( int i = 0; i < size; i ++ )
      if( ! operator[]( i ). save( file ) ) return false;
   return true;
#endif
}

template< typename T >
bool tnlList< T >::Load( tnlFile& file )
{
#ifdef HAVE_NOT_CXX11
   reset();
   int _size;
   file. read< int, tnlHost >( &_size );
   if( _size < 0 )
   {
      std::cerr << "The curve size is negative." << std::endl;
      return false;
   }
   T t;
   for( int i = 0; i < _size; i ++ )
   {
      if( ! file. read< T, tnlHost >( &t ) )
         return false;
      Append( t );
   }
   return true;
#else
   reset();
   int _size;
   file. read( &_size, 1 );
   if( _size < 0 )
   {
      std::cerr << "The curve size is negative." << std::endl;
      return false;
   }
   T t;
   for( int i = 0; i < _size; i ++ )
   {
      if( ! file. read( &t, 1 ) )
         return false;
      Append( t );
   }
   return true;
#endif
};

template< typename T >
bool tnlList< T >::DeepLoad( tnlFile& file )
{
#ifdef HAVE_NOT_CXX11
   reset();
   int _size;
   file. read< int, tnlHost >( &_size );
   if( _size < 0 )
   {
      std::cerr << "The list size is negative." << std::endl;
      return false;
   }
   for( int i = 0; i < _size; i ++ )
   {
      T t;
      if( ! t. load( file ) ) return false;
      Append( t );
   }
   return true;
#else
   reset();
   int _size;
   file. read( &_size );
   if( _size < 0 )
   {
      std::cerr << "The list size is negative." << std::endl;
      return false;
   }
   for( int i = 0; i < _size; i ++ )
   {
      T t;
      if( ! t. load( file ) ) return false;
      Append( t );
   }
   return true;
#endif
};
 
template< typename T >
std::ostream& operator << ( std::ostream& str, const tnlList< T >& list )
{
   int i, size( list. getSize() );
   for( i = 0; i < size; i ++ )
      str << "Item " << i << ":" << list[ i ] << std::endl;
   return str;
};

} // namespace TNL


