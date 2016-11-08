/***************************************************************************
                          List.h  -  description
                             -------------------
    begin                : Sat, 10 Apr 2004 15:58:51 +0100
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <stdlib.h>
#include <iostream>
#include <TNL/String.h>
#include <TNL/param-types.h>

namespace TNL {

class File;
template< class T > class DataElement;

//! Template for double linked lists
/*! To acces elements in the list one can use method getSize() and
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
template< class T > class List
{

   public:

      typedef T ElementType;

      //! Basic constructor
      List();

      //! Copy constructor
      List( const List& list );

      //! Destructor
      ~List();

      static String getType();

      //! If the list is empty return 'true'
      bool isEmpty() const;

      //! Return size of the list
      int getSize() const;

      //! Indexing operator
      T& operator[] ( const int& ind );

      //! Indexing operator for constant instances
      const T& operator[] ( const int& ind ) const;

      const List& operator = ( const List& lst );

      //! Append new data element
      bool Append( const T& data );

      //! Prepend new data element
      bool Prepend( const T& data );

      //! Insert new data element at given position
      bool Insert( const T& data, const int& ind );

      //! Append copy of another list
      bool AppendList( const List< T >& lst );

      //! Prepend copy of another list
      bool PrependList( const List< T >& lst );

      template< typename Array >
      void toArray( Array& array );

      //! Erase data element at given position
      void Erase( const int& ind );

      //! Erase data element with contained data at given position
      void DeepErase( const int& ind );

      //! Erase all data elements
      void reset();

      //! Erase all data elements with contained data
      void DeepEraseAll();

      //! Save the list in binary format
      bool Save( File& file ) const;

      //! Save the list in binary format using method save of type T
      bool DeepSave( File& file ) const;

      //! Load the list
      bool Load( File& file );

      //! Load the list using method Load of the type T
      bool DeepLoad( File& file );
 
   protected:

      //! Pointer to the first element
      DataElement< T >* first;

      //! Pointer to the last element
      /*! We use pointer to last element while adding new element to keep order of elements
       */
      DataElement< T >* last;

      //! List size
      int size;

      //! Iterator
      mutable DataElement< T >* iterator;

      //! Iterator index
      mutable int index;
 

};

template< typename T > std::ostream& operator << ( std::ostream& str, const List< T >& list );

//! Data element for List and mStack
template< class T > class DataElement
{
   //! Main data
   T data;

   //! Pointer to the next element
   DataElement< T >* next;

   //! Pointer to the previous element
   DataElement< T >* previous;

   public:
   //! Basic constructor
   DataElement()
      : next( 0 ),
        previous( 0 ){};

   //! Constructor with given data and possibly pointer to next element
   DataElement( const T& dt,
                   DataElement< T >* prv = 0,
                   DataElement< T >* nxt = 0 )
      : data( dt ),
        next( nxt ),
        previous( prv ){};

   //! Destructor
   ~DataElement(){};

   //! Return data for non-const instances
   T& Data() { return data; };

   //! Return data for const instances
   const T& Data() const { return data; };

   //! Return pointer to the next element for non-const instances
   DataElement< T >*& Next() { return next; };

   //! Return pointer to the next element for const instances
   const DataElement< T >* Next() const { return next; };

   //! Return pointer to the previous element for non-const instances
   DataElement< T >*& Previous() { return previous; };

   //! Return pointer to the previous element for const instances
   const DataElement< T >* Previous() const { return previous; };

};

} // namespace TNL

#include <TNL/List_impl.h>
