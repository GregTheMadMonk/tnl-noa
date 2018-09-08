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

template< int Size, typename Value >
class StaticArray
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = Size };

   __cuda_callable__
   inline StaticArray();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ Size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Value& v );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< Size, Value >& v );

   static String getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Value* getData();

   __cuda_callable__
   inline const Value* getData() const;

   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   __cuda_callable__
   inline Value& operator[]( int i );

   __cuda_callable__
   inline StaticArray< Size, Value >& operator = ( const StaticArray< Size, Value >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< Size, Value >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< Size, OtherValue >() const;

   __cuda_callable__
   inline void setValue( const ValueType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ Size ];
};

template< typename Value >
class StaticArray< 1, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 1 };

   __cuda_callable__
   inline StaticArray();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   __cuda_callable__
   inline StaticArray( const Value& v );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   static String getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Value* getData();

   __cuda_callable__
   inline const Value* getData() const;

   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   __cuda_callable__
   inline Value& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Value& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Value& x() const;

   __cuda_callable__
   inline StaticArray< 1, Value >& operator = ( const StaticArray< 1, Value >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 1, Value >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 1, OtherValue >() const;

   __cuda_callable__
   inline
   void setValue( const ValueType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

template< typename Value >
class StaticArray< 2, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 2 };

   __cuda_callable__
   inline StaticArray();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Value& v );

   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2 );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   static String getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Value* getData();

   __cuda_callable__
   inline const Value* getData() const;

   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   __cuda_callable__
   inline Value& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Value& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Value& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   inline Value& y();

   //! Returns the second coordinate
   __cuda_callable__
   inline const Value& y() const;

   __cuda_callable__
   inline StaticArray< 2, Value >& operator = ( const StaticArray< 2, Value >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 2, Value >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 2, OtherValue >() const;
 
   __cuda_callable__
   inline void setValue( const ValueType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

template< typename Value >
class StaticArray< 3, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 3 };

   __cuda_callable__
   inline StaticArray();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   //! This sets all vector components to v
   __cuda_callable__
   inline StaticArray( const Value& v );

   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2, const Value& v3 );

   //! Copy constructor
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   static String getType();

   __cuda_callable__
   inline int getSize() const;

   __cuda_callable__
   inline Value* getData();

   __cuda_callable__
   inline const Value* getData() const;

   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   __cuda_callable__
   inline Value& operator[]( int i );

   //! Returns the first coordinate
   __cuda_callable__
   inline Value& x();

   //! Returns the first coordinate
   __cuda_callable__
   inline const Value& x() const;

   //! Returns the second coordinate
   __cuda_callable__
   inline Value& y();

   //! Returns the second coordinate
   __cuda_callable__
   inline const Value& y() const;

   //! Returns the third coordinate
   __cuda_callable__
   inline Value& z();

   //! Returns the third coordinate
   __cuda_callable__
   inline const Value& z() const;

   __cuda_callable__
   inline StaticArray< 3, Value >& operator = ( const StaticArray< 3, Value >& array );

   template< typename Array >
   __cuda_callable__
   inline StaticArray< 3, Value >& operator = ( const Array& array );

   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;
 
   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 3, OtherValue >() const;

   __cuda_callable__
   inline void setValue( const ValueType& val );

   bool save( File& file ) const;

   bool load( File& file);

   void sort();
 
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

template< int Size, typename Value >
std::ostream& operator << ( std::ostream& str, const StaticArray< Size, Value >& a );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticArray_impl.h>
#include <TNL/Containers/StaticArray1D_impl.h>
#include <TNL/Containers/StaticArray2D_impl.h>
#include <TNL/Containers/StaticArray3D_impl.h>
