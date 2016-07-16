/***************************************************************************
                          tnlDataElement.h  -  description
                             -------------------
    begin                : 2004/04/11 14:01
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef __mDATAELEMENT_H__
#define __mDATAELEMENT_H__

//! Data element for tnlList and mStack
template< class T > class tnlDataElement
{
   //! Main data
   T data;

   //! Pointer to the next element
   tnlDataElement< T >* next;

   //! Pointer to the previous element
   tnlDataElement< T >* previous;

   public:
   //! Basic constructor
   tnlDataElement()
      : next( 0 ),
        previous( 0 ){};

   //! Constructor with given data and possibly pointer to next element
   tnlDataElement( const T& dt,
                   tnlDataElement< T >* prv = 0,
                   tnlDataElement< T >* nxt = 0 )
      : data( dt ),
        next( nxt ),
        previous( prv ){};

   //! Destructor
   ~tnlDataElement(){};

   //! Return data for non-const instances
   T& Data() { return data; };

   //! Return data for const instances
   const T& Data() const { return data; };

   //! Return pointer to the next element for non-const instances
   tnlDataElement< T >*& Next() { return next; };

   //! Return pointer to the next element for const instances
   const tnlDataElement< T >* Next() const { return next; };

   //! Return pointer to the previous element for non-const instances
   tnlDataElement< T >*& Previous() { return previous; };

   //! Return pointer to the previous element for const instances
   const tnlDataElement< T >* Previous() const { return previous; };

};

#endif
