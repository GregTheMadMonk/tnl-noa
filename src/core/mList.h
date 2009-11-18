/***************************************************************************
                          mList.h  -  description
                             -------------------
    begin                : Sat, 10 Apr 2004 15:58:51 +0100
    copyright            : (C) 2004 by Tom� Oberhuber
    email                : tomasoberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/**
  *@author Tom� Oberhuber
  */

#ifndef mListH
#define mListH

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "mDataElement.h"
#include "tnlString.h"
#include "param-types.h"

using namespace :: std;

//! Template for double linked lists
/*! To acces elements in the list one can use method Size() and
    operator[](). To add elements there are methods Append(), 
    Prepend() and Insert() to insert an element at given
    position. To erase particular element there is merthod
    Erase() taking the element position. To erase all elements
    there is method EraseAll. There are also alternatives DeepErase()
    and DeepEraseAll() to free dynamicaly allocated data inside the
    data elements.
    The list stores pointer to last accesed element so if one goes
    seqeuntialy through the list there is no inefficiency. The
    accesing algorithm is also able to deside whether to start from
    the last accesed position or from the begining resp. from the end
    of the list. So with common use one does not need to worry about
    efficiency :-)
 */
template< class T > class mList
{
   protected:

   //! Pointer to the first element
   mDataElement< T >* first;

   //! Pointer to the last element
   /*! We use pointer to last element while adding new element to keep order of elements
    */
   mDataElement< T >* last;
   
   //! List size
   long int size;

   //! Iterator
   mutable mDataElement< T >* iterator;

   //! Iterator index
   mutable long int index;

   public:
   //! Basic constructor
   mList() 
      : first( 0 ),
        last( 0 ),
        size( 0 ),
        iterator( 0 ),
        index( 0 ){};

   //! Copy constructor
   mList( const mList& list )
      : first( 0 ),
        last( 0 ),
        size( 0 ),
        iterator( 0 ),
        index( 0 )
   {
      AppendList( list );
   };

   //! Destructor
   ~mList() { EraseAll(); };

   //! If the list is empty return 'true'
   bool IsEmpty() const { return ! size; };
   
   //! Return size of the list
   long int Size() const { return size; };

   //! Indexing operator
   T& operator[] ( long int ind )
   {
      assert( ind < size );
      //if( ! size ) return NULL;
      // find fastest way to element with index i
      // we can use iterator as it is set now or
      // we can start from the first ro from the last
      // element
      //cout << "List operator[]: size = " << size
      //     << " current index = " << index
      //     << " index = " << ind << endl;
      long int iter_dist = abs( index - ind );
      if( ! iterator ||
          iter_dist > ind || 
          iter_dist > size - ind )
      {
         // it is better to start from the first one or the last one
         if( ind < size - ind )
         {
            // start from the first one
            //cout << "Setting curent index to 0." << endl;
            index = 0;
            iterator = first;
         }
         else
         {
            //cout << "Setting curent index to size - 1." << endl;
            index = size - 1;
            iterator = last;
         }
      }
      while( index != ind )
      {
         //cout << " current index = " << index
         //     << " index = " << ind << endl;
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
         assert( iterator );
      }
      return iterator -> Data();
   };
   
   //! Indexing operator for constant instances
   const T& operator[] ( long int ind ) const
   {
      return const_cast< mList< T >* >( this ) -> operator[]( ind );
   }

   //! Append new data element
   bool Append( const T& data )
   {
      if( ! first )
      {
         assert( ! last );
         first = last = new mDataElement< T >( data );
         if( ! first ) return false;
      }
      else 
      {
         mDataElement< T >* new_element =  new mDataElement< T >( data, last, 0 );
         if( ! new_element ) return false;
         assert( last );
         last = last -> Next() = new_element;
      }
      size ++;
      return true;
   };

   //! Prepend new data element
   bool Prepend( const T& data )
   {
      if( ! first )
      {
         assert( ! last );
         first = last = new mDataElement< T >( data );
         if( ! first ) return false;
      }
      else
      {
         mDataElement< T >* new_element =  new mDataElement< T >( data, 0, first );
         if( ! new_element ) return false;
         first = first -> Previous() = new_element; 
      }
      size ++;
      index ++;
      return true;
   };

   //! Insert new data element at given position
   bool Insert( const T& data, long int ind )
   {
      assert( ind <= size || ! size );
      if( ind == 0 ) return Prepend( data );
      if( ind == size ) return Append( data );
      operator[]( ind );
      mDataElement< T >* new_el = 
         new mDataElement< T >( data,
                                iterator -> Previous(),
                                iterator );
      if( ! new_el ) return false;
      iterator -> Previous() -> Next() = new_el;
      iterator -> Previous() = new_el;
      iterator = new_el;
      size ++;
      return true;
   };

   //! Append copy of another list
   bool AppendList( const mList< T >& lst )
   {
      long int i;
      for( i = 0; i < lst. Size(); i ++ )
      {
         if( ! Append( lst[ i ] ) ) return false;
      }
      return true;
   };
   
   //! Prepend copy of another list
   bool PrependList( const mList< T >& lst )
   
   {
      long int i;
      for( i = lst. Size(); i > 0; i -- )
         if( ! Prepend( lst[ i - 1 ] ) ) return false;
      return true;
   };

   //! Erase data element at given position
   void Erase( long int ind )
   {
      operator[]( ind );
      mDataElement< T >* tmp_it = iterator;
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

   //! Erase data element with contained data at given position
   void DeepErase( long int ind )
   {
      operator[]( ind );
      delete iterator -> Data();
      Erase( ind );
   };

   //! Erase all data elements
   void EraseAll()
   {
      iterator = first;
      mDataElement< T >* tmp_it;
      while( iterator )
      {
         tmp_it = iterator;
         iterator = iterator -> Next();
         delete tmp_it;
      }
      first = last = 0;
      size = 0;
   };

   //! Erase all data elements with contained data
   void DeepEraseAll()
   {
      iterator = first;
      mDataElement< T >* tmp_it;
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
   
   //! Save the list in binary format
   bool Save( ostream& file ) const
   {
      file. write( ( char* ) &size, sizeof( long int ) );
      long int i;
      for( i = 0; i < size; i ++ )
         file. write( ( char* ) & operator[]( i ), sizeof( T ) );
      if( file. bad() ) return false;
      return true;
   }

   //! Save the list in binary format using method Save of type T
   bool DeepSave( ostream& file ) const
   {
      file. write( ( char* ) &size, sizeof( long int ) );
      long int i;
      for( i = 0; i < size; i ++ )
         if( ! operator[]( i ). Save( file ) ) return false;
      if( file. bad() ) return false;
      return true;
   }

   //! Load the list
   bool Load( istream& file ) 
   {
      EraseAll();
      long int _size;
      file. read( ( char* ) &_size, sizeof( long int ) );
      if( _size < 0 )
      {
         cerr << "The curve size is negative." << endl;
         return false;
      }
      long int i;
      T t;
      for( i = 0; i < _size; i ++ )
      {
         file. read( ( char* ) &t, sizeof( T ) );
         Append( t );
      }
      if( file. bad() ) return false;
      return true;
   };

   //! Load the list using method Load of the type T
   bool DeepLoad( istream& file ) 
   {
      EraseAll();
      long int _size;
      file. read( ( char* ) &_size, sizeof( long int ) );
      if( _size < 0 )
      {
         cerr << "The list size is negative." << endl;
         return false;
      }
      long int i;
      for( i = 0; i < _size; i ++ )
      {
         T t;
         if( ! t. Load( file ) ) return false;
         Append( t );
      }
      if( file. bad() ) return false;
      return true;
   };
   
};

template< typename T > tnlString GetParameterType( const mList< T >& )
{
   T t;
   return tnlString( "mList< " ) + GetParameterType( t ) +  tnlString( " >" ); 
};

template< typename T > ostream& operator << ( ostream& str, const mList< T >& list )
{
   long int i, size( list. Size() );
   for( i = 0; i < size; i ++ )
      str << list[ i ] << endl;
   return str;
};

#endif
