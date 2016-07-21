/***************************************************************************
                          Array.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Object.h>
#include <TNL/Arrays/SharedArray.h>

// Forward declarations
namespace TNL {
class File;
class tnlHost;
} // namespace TNL

namespace TNL {
namespace Arrays {


template< typename Element, typename Device, typename Index >
class tnlSharedArray;

/****
 * Array handles memory allocation and sharing of the same data between more Arrays.
 *
 */
template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class Array : public virtual Object
{
   public:

      typedef Element ElementType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Arrays::Array< Element, tnlHost, Index > HostType;
      typedef Arrays::Array< Element, tnlCuda, Index > CudaType;
      typedef Arrays::Array< Element, Device, Index > ThisType;
 
      Array();
 
      Array( const IndexType& size );
 
      Array( Element* data,
                const IndexType& size );

      Array( Array< Element, Device, Index >& array,
                const IndexType& begin = 0,
                const IndexType& size = 0 );

      static String getType();

      String getTypeVirtual() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      /****
       * This sets size of the array. If the array shares data with other arrays
       * these data are released. If the current data are not shared and the current
       * size is the same as the new one, nothing happens.
       */
      bool setSize( Index size );

      template< typename Array >
      bool setLike( const Array& array );

      void bind( Element* _data,
                 const Index _size );

      void bind( const Array< Element, Device, Index >& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( tnlStaticArray< Size, Element >& array );

      void swap( Array< Element, Device, Index >& array );

      void reset();

      __cuda_callable__ Index getSize() const;

      void setElement( const Index& i, const Element& x );

      Element getElement( const Index& i ) const;

      __cuda_callable__ inline Element& operator[] ( const Index& i );

      __cuda_callable__ inline const Element& operator[] ( const Index& i ) const;

      Array< Element, Device, Index >& operator = ( const Array< Element, Device, Index >& array );

      template< typename ArrayT >
      Array< Element, Device, Index >& operator = ( const ArrayT& array );

      template< typename ArrayT >
      bool operator == ( const ArrayT& array ) const;

      template< typename ArrayT >
      bool operator != ( const ArrayT& array ) const;

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

      //! Method for loading the object from a file as a binary data.
      bool load( File& file );
 
      //! This method loads data without reallocation.
      /****
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       */
      bool boundLoad( File& file );
 
      bool boundLoad( const String& fileName );
 
      using Object::load;

      using Object::save;

      ~Array();

   protected:
 
      void releaseData() const;

      //!Number of elements in array
      mutable Index size;

      //! Pointer to data
      mutable Element* data;

      /****
       * Pointer to the originally allocated data. They might differ if one
       * long array is partitioned into more shorter arrays. Each of them
       * must know the pointer on allocated data because the last one must
       * deallocate the array. If outer data (not allocated by TNL) are bind
       * then this pointer is zero since no deallocation is necessary.
       */
      mutable Element* allocationPointer;

      /****
       * Counter of objects sharing this array or some parts of it. The reference counter is
       * allocated after first sharing of the data between more arrays. This is to avoid
       * unnecessary dynamic memory allocation.
       */
      mutable int* referenceCounter;
};

template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Array< Element, Device, Index >& v );

} // namespace Arrays
} // namespace TNL

#include <TNL/Arrays/Array_impl.h>

