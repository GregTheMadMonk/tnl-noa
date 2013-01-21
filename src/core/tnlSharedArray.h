/***************************************************************************
                          tnlSharedArray.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSHAREDARRAY_H_
#define TNLSHAREDARRAY_H_

#include <core/tnlObject.h>

class tnlFile;
class tnlHost;

template< typename Element, typename Device, typename Index >
class tnlArray;

template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class tnlSharedArray : public tnlObject
{
   public:

   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlSharedArray();

   tnlString getType() const;

   void bind( Element* _data,
              const Index _size );

   void bind( tnlArray< Element, Device, Index >& array );

   void bind( tnlSharedArray< Element, Device, Index >& array );

   void swap( tnlSharedArray< Element, Device, Index >& array );

   void reset();

   Index getSize() const;

   void setElement( const Index i, const Element& x );

   Element getElement( Index i ) const;

   Element& operator[] ( Index i );

   const Element& operator[] ( Index i ) const;

   tnlSharedArray< Element, Device, Index >& operator = ( const tnlSharedArray< Element, Device, Index >& array );

   template< typename Array >
   tnlSharedArray< Element, Device, Index >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

   void setValue( const Element& e );

   const Element* getData() const;

   Element* getData();

   /*!
    * Returns true if non-zero size is set.
    */
   operator bool() const;

   //! This method measures data transfers done by this vector.
   /*!
    * Every time one touches this grid touches * size * sizeof( Real ) bytes are added
    * to transfered bytes in tnlStatistics.
    */
   template< typename IndexType2 = Index >
   void touch( IndexType2 touches = 1 ) const;

   //! Method for saving the object to a file as a binary data.
   bool save( tnlFile& file ) const;

   bool save( const tnlString& fileName ) const;

   protected:

   //!Number of allocated elements
   Index size;

   //! Pointer to allocated data
   Element* data;
};

#include <implementation/core/tnlSharedArray_impl.h>

#endif /* TNLSHAREDARRAY_H_ */
