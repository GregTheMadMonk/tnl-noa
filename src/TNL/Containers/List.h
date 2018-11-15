/***************************************************************************
                          List.h  -  description
                             -------------------
    begin                : Sat, 10 Apr 2004 15:58:51 +0100
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/Assert.h>
#include <TNL/File.h>
#include <TNL/String.h>
#include <TNL/param-types.h>

namespace TNL {
namespace Containers {

template< class T > class ListDataElement;

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
      typedef T ValueType;

      /// \brief Basic constructor.
      ///
      /// Constructs an empty list.
      List();

      /// \brief Copy constructor.
      ///
      /// Construct a copy of \e list.
      /// @param list Name of another list.
      List( const List& list );

      /// \brief Destructor.
      ///
      /// Destroys the list. References to the values in the list become invalid.
      ~List();

      static String getType();

      /// Returns \e true if the list contains no items, otherwise returns \e false.
      bool isEmpty() const;

      /// Returns number of items in the list.
      int getSize() const;

      /// Indexing operator.
      T& operator[] ( const int& ind );

      /// Indexing operator for constant instances.
      const T& operator[] ( const int& ind ) const;

      const List& operator = ( const List& lst );

      bool operator == ( const List& lst ) const;

      bool operator != ( const List& lst ) const;

      /// \brief Appends new data element.
      ///
      /// Inserts \e data at the end of the list.
      bool Append( const T& data );

      /// \brief Prepends new data element.
      ///
      /// Inserts \e data at the beginning of the list.
      bool Prepend( const T& data );

      /// \brief Inserts new data element at given position.
      ///
      /// Inserts \e data at index position \e ind in the list.
      bool Insert( const T& data, const int& ind );

      /// Appends copy of another list.
      bool AppendList( const List< T >& lst );

      /// Prepends copy of another list.
      bool PrependList( const List< T >& lst );

      template< typename Array >
      void toArray( Array& array );

      /// Erases data element at given position.
      void Erase( const int& ind );

      /// Erases data element with contained data at given position.
      void DeepErase( const int& ind );

      /// Erases all data elements.
      void reset();

      /// Erases all data elements with contained data.
      void DeepEraseAll();

      /// Saves the list in binary format.
      bool Save( File& file ) const;

      /// Saves the list in binary format using method save of type T.
      bool DeepSave( File& file ) const;

      /// Loads the list.
      bool Load( File& file );

      /// Loads the list using method Load of the type T.
      bool DeepLoad( File& file );
 
   protected:
      /// Pointer to the first element.
      ListDataElement< T >* first;

      /// Pointer to the last element.
      /*! We use pointer to last element while adding new element to keep order of elements
       */
      ListDataElement< T >* last;

      /// List size.
      int size;

      /// Iterator.
      mutable ListDataElement< T >* iterator;

      /// Iterator index.
      mutable int index;
};

template< typename T > std::ostream& operator << ( std::ostream& str, const List< T >& list );

//! Data element for List and mStack
template< class T > class ListDataElement
{
   //! Main data
   T data;

   //! Pointer to the next element
   ListDataElement< T >* next;

   //! Pointer to the previous element
   ListDataElement< T >* previous;

   public:
   //! Basic constructor
   ListDataElement()
      : next( 0 ),
        previous( 0 ){};

   //! Constructor with given data and possibly pointer to next element
   ListDataElement( const T& dt,
                    ListDataElement< T >* prv = 0,
                    ListDataElement< T >* nxt = 0 )
      : data( dt ),
        next( nxt ),
        previous( prv ){};

   //! Destructor
   ~ListDataElement(){};

   //! Return data for non-const instances
   T& Data() { return data; };

   //! Return data for const instances
   const T& Data() const { return data; };

   //! Return pointer to the next element for non-const instances
   ListDataElement< T >*& Next() { return next; };

   //! Return pointer to the next element for const instances
   const ListDataElement< T >* Next() const { return next; };

   //! Return pointer to the previous element for non-const instances
   ListDataElement< T >*& Previous() { return previous; };

   //! Return pointer to the previous element for const instances
   const ListDataElement< T >* Previous() const { return previous; };
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/List_impl.h>
