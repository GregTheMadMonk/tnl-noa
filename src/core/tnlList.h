/***************************************************************************
                          tnlList.h  -  description
                             -------------------
    begin                : Sat, 10 Apr 2004 15:58:51 +0100
    copyright            : (C) 2004 by Tomas Oberhuber
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

#ifndef mListH
#define mListH

#include <core/tnlAssert.h>
#include <stdlib.h>
#include <iostream>
#include <core/tnlDataElement.h>
#include <core/tnlString.h>

#include <core/param-types.h>

class tnlFile;

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
template< class T > class tnlList
{

   public:

      typedef T ElementType;

      //! Basic constructor
      tnlList();

      //! Copy constructor
      tnlList( const tnlList& list );

      //! Destructor
      ~tnlList();

      static tnlString getType();

      //! If the list is empty return 'true'
      bool isEmpty() const;

      //! Return size of the list
      int getSize() const;

      //! Indexing operator
      T& operator[] ( const int& ind );

      //! Indexing operator for constant instances
      const T& operator[] ( const int& ind ) const;

      const tnlList& operator = ( const tnlList& lst );

      //! Append new data element
      bool Append( const T& data );

      //! Prepend new data element
      bool Prepend( const T& data );

      //! Insert new data element at given position
      bool Insert( const T& data, const int& ind );

      //! Append copy of another list
      bool AppendList( const tnlList< T >& lst );

      //! Prepend copy of another list
      bool PrependList( const tnlList< T >& lst );   

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
      bool Save( tnlFile& file ) const;

      //! Save the list in binary format using method save of type T
      bool DeepSave( tnlFile& file ) const;

      //! Load the list
      bool Load( tnlFile& file );

      //! Load the list using method Load of the type T
      bool DeepLoad( tnlFile& file );
   
   protected:

      //! Pointer to the first element
      tnlDataElement< T >* first;

      //! Pointer to the last element
      /*! We use pointer to last element while adding new element to keep order of elements
       */
      tnlDataElement< T >* last;

      //! List size
      int size;

      //! Iterator
      mutable tnlDataElement< T >* iterator;

      //! Iterator index
      mutable int index;   
      

};

template< typename T > ostream& operator << ( ostream& str, const tnlList< T >& list );

#include<core/tnlList_impl.h>

#endif
