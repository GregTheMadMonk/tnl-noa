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

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element v[ Size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const tnlStaticArray< Size, Element >& v );

   static tnlString getType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   int getSize() const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& operator[]( int i ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& operator[]( int i );

   tnlStaticArray< Size, Element >& operator = ( const tnlStaticArray< Size, Element >& array );

   template< typename Array >
   tnlStaticArray< Size, Element >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

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

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element v[ size ] );

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   int getSize() const;


#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& operator[]( int i ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& operator[]( int i );

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& x();

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& x() const;

   tnlStaticArray< 1, Element >& operator = ( const tnlStaticArray< 1, Element >& array );

   template< typename Array >
   tnlStaticArray< 1, Element >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

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

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v );

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v1, const Element& v2 );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   int getSize() const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& operator[]( int i ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& operator[]( int i );

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& x();

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& x() const;

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& y();

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& y() const;

   tnlStaticArray< 2, Element >& operator = ( const tnlStaticArray< 2, Element >& array );

   template< typename Array >
   tnlStaticArray< 2, Element >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

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

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v );

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const Element& v1, const Element& v2, const Element& v3 );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   int getSize() const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& operator[]( int i ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& operator[]( int i );

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& x();

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& x() const;

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& y();

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& y() const;

   //! Returns the third coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Element& z();

   //! Returns the third coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Element& z() const;

   tnlStaticArray< 3, Element >& operator = ( const tnlStaticArray< 3, Element >& array );

   template< typename Array >
   tnlStaticArray< 3, Element >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();

   protected:
   Element data[ size ];

};

template< int Size, typename Element >
ostream& operator << ( ostream& str, const tnlStaticArray< Size, Element >& a );

#include <implementation/core/arrays/tnlStaticArray_impl.h>
#include <implementation/core/arrays/tnlStaticArray1D_impl.h>
#include <implementation/core/arrays/tnlStaticArray2D_impl.h>
#include <implementation/core/arrays/tnlStaticArray3D_impl.h>

#endif /* TNLSTATICARRAY_H_ */
