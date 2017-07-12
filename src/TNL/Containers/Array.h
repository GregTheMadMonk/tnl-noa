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
#include <TNL/File.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Containers {

template< int, typename > class StaticArray;

/****
 * Array handles memory allocation and sharing of the same data between more Arrays.
 *
 */
template< typename Element,
          typename Device = Devices::Host,
          typename Index = int >
class Array : public virtual Object
{
   public:

      typedef Element ElementType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Containers::Array< Element, Devices::Host, Index > HostType;
      typedef Containers::Array< Element, Devices::Cuda, Index > CudaType;

      Array();

      Array( const IndexType& size );

      Array( Element* data,
             const IndexType& size );

      Array( Array& array,
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
      void setSize( Index size );

      __cuda_callable__ Index getSize() const;

      template< typename ArrayT >
      void setLike( const ArrayT& array );

      void bind( Element* _data,
                 const Index _size );

      template< typename ArrayT >
      void bind( const ArrayT& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( StaticArray< Size, Element >& array );

      void swap( Array& array );

      void reset();

      void setElement( const Index& i, const Element& x );

      Element getElement( const Index& i ) const;

      __cuda_callable__ inline Element& operator[] ( const Index& i );

      __cuda_callable__ inline const Element& operator[] ( const Index& i ) const;

      Array& operator = ( const Array& array );

      template< typename ArrayT >
      Array& operator = ( const ArrayT& array );

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
      template< typename IndexType2 = Index >
      void touch( IndexType2 touches = 1 ) const;

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

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Array_impl.h>

