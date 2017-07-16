/***************************************************************************
                          StaticArray.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/String.h>
#include <TNL/File.h>

namespace TNL {
namespace Containers {   

template< int Size, typename Element >
class StaticArray
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = Size };

   __cuda_callable__
   inline StaticArray();

   __cuda_callable__
   inline StaticArray( const Element v[ Size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< Size, Element >& v );

   static String getType();

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
   inline StaticArray< Size, Element >& operator = ( const StaticArray< Size, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< Size, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator StaticArray< Size, OtherElement >() const;

   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Element data[ Size ];
};

template< typename Element >
class StaticArray< 1, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 1 };

   __cuda_callable__
   inline StaticArray();

   __cuda_callable__
   inline StaticArray( const Element v[ size ] );

   __cuda_callable__
   inline StaticArray( const Element& v );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Element >& v );

   static String getType();

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
   inline StaticArray< 1, Element >& operator = ( const StaticArray< 1, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 1, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator StaticArray< 1, OtherElement >() const;

   __cuda_callable__
   inline
   void setValue( const ElementType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Element data[ size ];
};

template< typename Element >
class StaticArray< 2, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 2 };

   __cuda_callable__
   inline StaticArray();

   __cuda_callable__
   inline StaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Element& v );

   __cuda_callable__
   inline StaticArray( const Element& v1, const Element& v2 );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Element >& v );

   static String getType();

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
   inline StaticArray< 2, Element >& operator = ( const StaticArray< 2, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 2, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator StaticArray< 2, OtherElement >() const;
 
   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Element data[ size ];
};

template< typename Element >
class StaticArray< 3, Element >
{
   public:
   typedef Element ElementType;
   typedef int     IndexType;
   enum { size = 3 };

   __cuda_callable__
   inline StaticArray();

   __cuda_callable__
   inline StaticArray( const Element v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Element& v );

   __cuda_callable__
   inline StaticArray( const Element& v1, const Element& v2, const Element& v3 );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Element >& v );

   static String getType();

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
   inline StaticArray< 3, Element >& operator = ( const StaticArray< 3, Element >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 3, Element >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherElement >
   __cuda_callable__
   operator StaticArray< 3, OtherElement >() const;

   __cuda_callable__
   inline void setValue( const ElementType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Element data[ size ];
};

template< int Size, typename Element >
std::ostream& operator << ( std::ostream& str, const StaticArray< Size, Element >& a );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticArray_impl.h>
#include <TNL/Containers/StaticArray1D_impl.h>
#include <TNL/Containers/StaticArray2D_impl.h>
#include <TNL/Containers/StaticArray3D_impl.h>
