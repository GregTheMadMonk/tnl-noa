/***************************************************************************
                          mDataElement.h  -  description
                             -------------------
    begin                : 2004/04/11 14:01
    copyright            : (C) 2004 by Tomas Oberhuber
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

#ifndef __mDATAELEMENT_H__
#define __mDATAELEMENT_H__

//! Data element for tnlList and mStack
template< class T > class mDataElement
{
   //! Main data
   T data;

   //! Pointer to the next element
   mDataElement< T >* next;

   //! Pointer to the previous element
   mDataElement< T >* previous;

   public:
   //! Basic constructor
   mDataElement()
      : next( 0 ),
        previous( 0 ){};

   //! Constructor with given data and possibly pointer to next element
   mDataElement( const T& dt, 
                  mDataElement< T >* prv = 0,
                  mDataElement< T >* nxt = 0 )
      : data( dt ), 
        next( nxt ),
        previous( prv ){};

   //! Destructor
   ~mDataElement(){};

   //! Return data for non-const instances
   T& Data() { return data; };

   //! Return data for const instances
   const T& Data() const { return data; };

   //! Return pointer to the next element for non-const instances
   mDataElement< T >*& Next() { return next; };

   //! Return pointer to the next element for const instances
   const mDataElement< T >* Next() const { return next; };

   //! Return pointer to the previous element for non-const instances
   mDataElement< T >*& Previous() { return previous; };

   //! Return pointer to the previous element for const instances
   const mDataElement< T >* Previous() const { return previous; };

};

#endif
