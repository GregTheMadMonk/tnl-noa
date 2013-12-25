/***************************************************************************
                          tnlArray.h -  description
                             -------------------
    begin                : Jul 4, 2012
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

#ifndef TNLARRAY_H_
#define TNLARRAY_H_

#include <core/tnlObject.h>
#include <core/arrays/tnlSharedArray.h>

class tnlFile;
class tnlHost;

template< typename Element, typename Device, typename Index >
class tnlSharedArray;

template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class tnlArray : public virtual tnlObject
{
   public:

   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlArray();

   tnlArray( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setSize( Index size );

   template< typename Array >
   bool setLike( const Array& array );

   void swap( tnlArray< Element, Device, Index >& array );

   void reset();

   Index getSize() const;

   void setElement( const Index i, const Element& x );

   Element getElement( Index i ) const;

   Element& operator[] ( Index i );

   const Element& operator[] ( Index i ) const;

   tnlArray< Element, Device, Index >& operator = ( const tnlArray< Element, Device, Index >& array );

   template< typename Array >
   tnlArray< Element, Device, Index >& operator = ( const Array& array );

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
#ifdef HAVE_NOT_CXX11
   template< typename IndexType2 >
   void touch( IndexType2 touches = 1 ) const;
#else
   template< typename IndexType2 = Index >
   void touch( IndexType2 touches = 1 ) const;
#endif      

   //! Method for saving the object to a file as a binary data.
   bool save( tnlFile& file ) const;

   //! Method for loading the object from a file as a binary data.
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   ~tnlArray();

   protected:

   //!Number of allocated elements
   Index size;

   //! Pointer to allocated data
   Element* data;
};

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlArray< Element, Device, Index >& v );

#include <implementation/core/arrays/tnlArray_impl.h>

#endif /* TNLARRAY_H_ */
