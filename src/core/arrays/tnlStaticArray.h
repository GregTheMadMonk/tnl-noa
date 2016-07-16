/***************************************************************************
                          tnlStaticArray.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
   inline tnlStaticArray();

   __cuda_callable__
   inline tnlStaticArray( const Element v[ Size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline tnlStaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   inline tnlStaticArray( const tnlStaticArray< Size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Element* getData();

   __cuda_callable__
   inline const Element* getData() const;

   __cuda_callable__
   inline const Element& operator[]( int i ) const;

   __cuda_callable__
   inline Element& operator[]( int i );

   __cuda_callable__
   inline tnlStaticArray< Size, Element >& operator = ( const tnlStaticArray< Size, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline tnlStaticArray< Size, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator tnlStaticArray< Size, OtherElement >() const;

   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();
 
   ostream& write( ostream& str, const char* separator = " " ) const;

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
   inline tnlStaticArray();

   __cuda_callable__
   inline tnlStaticArray( const Element v[ size ] );

   __cuda_callable__
   inline tnlStaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   inline tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Element* getData();

   __cuda_callable__
   inline const Element* getData() const;

   __cuda_callable__
   inline const Element& operator[]( int i ) const;

   __cuda_callable__
   inline Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Element& x() const;

   __cuda_callable__
   inline tnlStaticArray< 1, Element >& operator = ( const tnlStaticArray< 1, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline tnlStaticArray< 1, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator tnlStaticArray< 1, OtherElement >() const;

   __cuda_callable__
   inline
   void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();
 
   ostream& write( ostream& str, const char* separator = " " ) const;

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
   inline tnlStaticArray();

   __cuda_callable__
   inline tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline tnlStaticArray( const Element& v );

   __cuda_callable__
   inline tnlStaticArray( const Element& v1, const Element& v2 );

   //! Copy constructor
   __cuda_callable__
   inline tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Element* getData();

   __cuda_callable__
   inline const Element* getData() const;

   __cuda_callable__
   inline const Element& operator[]( int i ) const;

   __cuda_callable__
   inline Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Element& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   inline Element& y();

   //! Returns the second coordinate
   __cuda_callable__
   inline const Element& y() const;

   __cuda_callable__
   inline tnlStaticArray< 2, Element >& operator = ( const tnlStaticArray< 2, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline tnlStaticArray< 2, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator tnlStaticArray< 2, OtherElement >() const;
 
   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();
 
   ostream& write( ostream& str, const char* separator = " " ) const;

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
   inline tnlStaticArray();

   __cuda_callable__
   inline tnlStaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline tnlStaticArray( const Element& v );

   __cuda_callable__
   inline tnlStaticArray( const Element& v1, const Element& v2, const Element& v3 );

   //! Copy constructor
   __cuda_callable__
   inline tnlStaticArray( const tnlStaticArray< size, Element >& v );

   static tnlString getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Element* getData();

   __cuda_callable__
   inline const Element* getData() const;

   __cuda_callable__
   inline const Element& operator[]( int i ) const;

   __cuda_callable__
   inline Element& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Element& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Element& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   inline Element& y();

   //! Returns the second coordinate
   __cuda_callable__
   inline const Element& y() const;

   //! Returns the third coordinate
   __cuda_callable__
   inline Element& z();

   //! Returns the third coordinate
   __cuda_callable__
   inline const Element& z() const;

   __cuda_callable__
   inline tnlStaticArray< 3, Element >& operator = ( const tnlStaticArray< 3, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline tnlStaticArray< 3, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator tnlStaticArray< 3, OtherElement >() const;

   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   void sort();
 
   ostream& write( ostream& str, const char* separator = " " ) const;

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
