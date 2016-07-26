/***************************************************************************
                          tnlSharedArray.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Object.h>
#include <TNL/core/tnlCuda.h>

// Forward declarations
namespace TNL {
   class File;
   class tnlHost;
   class tnlCuda;

namespace Arrays {   


template< typename Element, typename Device, typename Index >
class Array;

template< int Size, typename Element >
class tnlStaticArray;

template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class tnlSharedArray : public Object
{
   public:

   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlSharedArray< Element, tnlHost, Index > HostType;
   typedef tnlSharedArray< Element, tnlCuda, Index > CudaType;

   __cuda_callable__
   tnlSharedArray();

   __cuda_callable__
   tnlSharedArray( Element* _data,
                   const Index _size );

   __cuda_callable__
   tnlSharedArray( Array< Element, Device, Index >& array );

   __cuda_callable__
   tnlSharedArray( tnlSharedArray< Element, Device, Index >& array );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   __cuda_callable__
   void bind( Element* _data,
              const Index _size );

   template< typename Array >
   __cuda_callable__
   void bind( Array& array,
              IndexType index = 0,
              IndexType size = 0 );

   template< int Size >
   __cuda_callable__
   void bind( tnlStaticArray< Size, Element >& array );

   __cuda_callable__
   void bind( tnlSharedArray< Element, Device, Index >& array );

   void swap( tnlSharedArray< Element, Device, Index >& array );

   void reset();

   __cuda_callable__ Index getSize() const;

   void setElement( const Index& i, const Element& x );

   Element getElement( const Index& i ) const;

   __cuda_callable__ Element& operator[] ( const Index& i );

   __cuda_callable__ const Element& operator[] ( const Index& i ) const;

   tnlSharedArray< Element, Device, Index >& operator = ( const tnlSharedArray< Element, Device, Index >& array );

   template< typename Array >
   tnlSharedArray< Element, Device, Index >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

   void setValue( const Element& e );

   __cuda_callable__ const Element* getData() const;

   __cuda_callable__ Element* getData();

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
   bool save( File& file ) const;

   bool save( const String& fileName ) const;

   bool load( File& file );

   bool load( const String& fileName );

   protected:

   //!Number of allocated elements
   Index size;

   //! Pointer to allocated data
   Element* data;
};

template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const tnlSharedArray< Element, Device, Index >& v );

} // namespace Arrays
} // namespace TNL

#include <TNL/Arrays/SharedArray_impl.h>
