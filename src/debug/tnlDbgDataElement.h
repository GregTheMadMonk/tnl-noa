/***************************************************************************
                          tnlDbgDataElement.h  -  description
                             -------------------
    begin                : 2004/04/11 14:01
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

#ifndef __TNLDBGDATAELEMENT_H__
#define __TNLDBGDATAELEMENT_H__

template< class T> class dbgList;

//! Data element for dbgList
template< class T > class tnlDbgDataElement
{
   //! Main data
   T data;

   //! Pointer to the next element
   tnlDbgDataElement< T >* next;

   //! Pointer to the previous element
   tnlDbgDataElement< T >* previous;

   public:
   //! Basic constructor
   tnlDbgDataElement()
      : next( 0 ),
        previous( 0 ){};

   //! Constructor with given data and possibly pointer to next element
   tnlDbgDataElement( const T& dt, 
                      tnlDbgDataElement< T >* prv = 0,
                      tnlDbgDataElement< T >* nxt = 0 )
      : data( dt ), 
        next( nxt ),
        previous( prv ){};

   //! Destructor
   ~tnlDbgDataElement(){};

   //! Return data for non-const instances
   T& Data() { return data; };

   //! Return data for const instances
   const T& Data() const { return data; };

   //! Return pointer to the next element for non-const instances
   tnlDbgDataElement< T >*& Next() { return next; };

   //! Return pointer to the next element for const instances
   const tnlDbgDataElement< T >* Next() const { return next; };

   //! Return pointer to the previous element for non-const instances
   tnlDbgDataElement< T >*& Previous() { return previous; };

   //! Return pointer to the previous element for const instances
   const tnlDbgDataElement< T >* Previous() const { return previous; };

};

#endif
