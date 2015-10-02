/***************************************************************************
                          tnlStaticArray.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSTATICARRAY_H_
#define TNLSTATICARRAY_H_

#include <core/tnlString.h>
#include <core/tnlFile.h>

//! Aliases for the coordinates
// TODO: Remove this - it is here only because of some legact code
enum { tnlX = 0, tnlY, tnlZ };

template< int Size, typename Element >
class tnlStaticArray
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = Size };

   __cuda_callable__
   tnlStaticArray();

   __cuda_callable__
   tnlStaticArray( const Element v[ Size ] );

   //! This sets all vector components to v
   __cuda_callable__
   tnlStaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   tnlStaticArray( const tnlStaticArray< Size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   int getSize() const;

   __cuda_callable__
   Element* getData();

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   const Element& operator[]( int i ) const;

   __cuda_callable__
   Element& operator[]( int i );

   __cuda_callable__
   tnlStaticArray< Size, Element >& operator = ( const tnlStaticArray< Size, Element >& array );

   template< typename Array >
   __cuda_callable__
   tnlStaticArray< Size, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   bool operator != ( const Array& array ) const;

   void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();

   protected:
   Element data[ Size ];

};

template< typename Element >
class tnlStaticArray< 1, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 1 };

   __cuda_callable__
   tnlStaticArray();

   __cuda_callable__
   tnlStaticArray( const Element v[ size ] );

   __cuda_callable__
   tnlStaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   int getSize() const;

   __cuda_callable__
   Element* getData();

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   const Element& operator[]( int i ) const;

   __cuda_callable__
   Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   const Element& x() const;

   __cuda_callable__
   tnlStaticArray< 1, Element >& operator = ( const tnlStaticArray< 1, Element >& array );

   template< typename Array >
   __cuda_callable__
   tnlStaticArray< 1, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   bool operator != ( const Array& array ) const;

   void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();

   protected:
   Element data[ size ];
};

template< typename Element >
class tnlStaticArray< 2, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 2 };

   __cuda_callable__
   tnlStaticArray();

   __cuda_callable__
   tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   tnlStaticArray( const Element& v );

   __cuda_callable__
   tnlStaticArray( const Element& v1, const Element& v2 );

   //! Copy constructor
   __cuda_callable__
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   int getSize() const;

   __cuda_callable__
   Element* getData();

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   const Element& operator[]( int i ) const;

   __cuda_callable__
   Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   const Element& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   Element& y();

   //! Returns the second coordinate
   __cuda_callable__
   const Element& y() const;

   __cuda_callable__
   tnlStaticArray< 2, Element >& operator = ( const tnlStaticArray< 2, Element >& array );

   template< typename Array >
   __cuda_callable__
   tnlStaticArray< 2, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   bool operator != ( const Array& array ) const;

   void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();

   protected:
   Element data[ size ];
};

template< typename Element >
class tnlStaticArray< 3, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 3 };

   __cuda_callable__
   tnlStaticArray();

   __cuda_callable__
   tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   tnlStaticArray( const Element& v );

   __cuda_callable__
   tnlStaticArray( const Element& v1, const Element& v2, const Element& v3 );

   //! Copy constructor
   __cuda_callable__
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   int getSize() const;

   __cuda_callable__
   Element* getData();

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   const Element& operator[]( int i ) const;

   __cuda_callable__
   Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   const Element& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   Element& y();

   //! Returns the second coordinate
   __cuda_callable__
   const Element& y() const;

   //! Returns the third coordinate
   __cuda_callable__
   Element& z();

   //! Returns the third coordinate
   __cuda_callable__
   const Element& z() const;

   __cuda_callable__
   tnlStaticArray< 3, Element >& operator = ( const tnlStaticArray< 3, Element >& array );

   template< typename Array >
   __cuda_callable__
   tnlStaticArray< 3, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   bool operator != ( const Array& array ) const;

   void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();

   protected:
   Element data[ size ];

};

template< int Size, typename Element >
ostream& operator << ( ostream& str, const tnlStaticArray< Size, Element >& a );

#include <core/arrays/tnlStaticArray_impl.h>
#include <core/arrays/tnlStaticArray1D_impl.h>
#include <core/arrays/tnlStaticArray2D_impl.h>
#include <core/arrays/tnlStaticArray3D_impl.h>

#endif /* TNLSTATICARRAY_H_ */
