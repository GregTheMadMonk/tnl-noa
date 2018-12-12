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

/**
 * Array handles memory allocation and sharing of the same data between more Arrays.
 *
 * \tparam Value Type of array values.
 * \tparam Device Device type.
 * \tparam Index Type of index.
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
class Array : public Object
{
   public:

      typedef Value ValueType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Containers::Array< Value, Devices::Host, Index > HostType;
      typedef Containers::Array< Value, Devices::Cuda, Index > CudaType;

      /** \brief Basic constructor.
       *
       * Constructs an empty array with the size of zero.
       */
      Array();

      /**
       * \brief Constructor with size.
       *
       * \param size Number of array elements. / Size of allocated memory.
       */
      Array( const IndexType& size );

      /**
       * \brief Constructor with data and size.
       *
       * \param data
       * \param size Number of array elements.
       */
      Array( Value* data,
             const IndexType& size );

      /**
       * \brief Copy constructor.
       *
       * Copies \e size elements from existing \e array into a new array.
       * \param array Existing array that is about to be copied.
       * \param begin Index from which the array is copied.
       * \param size Number of array elements that should be copied.
       */
      Array( Array& array,
             const IndexType& begin = 0,
             const IndexType& size = 0 );

      /** \brief Returns type of array value, device type and the type of index. */
      static String getType();

      /** \brief Returns type of array value, device type and the type of index. */
      virtual String getTypeVirtual() const;

      /** \brief Returns (host) type of array value, device type and the type of index. */
      static String getSerializationType();

      /** \brief Returns (host) type of array value, device type and the type of index. */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for setting the size of an array.
       *
       * If the array shares data with other arrays these data are released.
       * If the current data are not shared and the current size is the same
       * as the new one, nothing happens.
       *
       * \param size Number of array elements.
       */
      void setSize( Index size );

      /** \brief Method for getting the size of an array. */
      __cuda_callable__ Index getSize() const;

      /**
       * \brief Assigns features of the existing \e array to the given array.
       *
       * Sets the same size as the size of existing \e array.
       * \tparam ArrayT Type of array.
       * \param array Existing array.
       */
      template< typename ArrayT >
      void setLike( const ArrayT& array );

      void bind( Value* _data,
                 const Index _size );

      template< typename ArrayT >
      void bind( const ArrayT& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( StaticArray< Size, Value >& array );

      /**
       * \brief Swaps all features of given array with existing \e array.
       *
       * Swaps sizes, all values (data), allocated memory and references of given
       * array with existing array.
       * \param array Existing array, which features are swaped with given array.
       */
      void swap( Array& array );

      /**
       * \brief Resets the given array.
       *
       * Releases all data from array.
       */
      void reset();

      /**
       * \brief Method for getting the data from given array with constant poiner.
       */
      __cuda_callable__ const Value* getData() const;

      /**
       * \brief Method for getting the data from given array.
       */
      __cuda_callable__ Value* getData();

      /**
       * \brief Assignes the value \e x to the array element at position \e i.
       *
       * \param i Index position.
       * \param x New value of an element.
       */
      void setElement( const Index& i, const Value& x );

      /**
       * \brief Accesses specified element at the position \e i and returns its value.
       *
       * \param i Index position of an element.
       */
      Value getElement( const Index& i ) const;

      /**
       * \brief Accesses specified element at the position \e i and returns a reference to its value.
       *
       * \param i Index position of an element.
       */
      __cuda_callable__ inline Value& operator[] ( const Index& i );

      /**
       * \brief Accesses specified element at the position \e i and returns a (constant?) reference to its value.
       *
       * \param i Index position of an element.
       */
      __cuda_callable__ inline const Value& operator[] ( const Index& i ) const;

      /** Assigns \e array to the given array, replacing its current contents. */
      Array& operator = ( const Array& array );

      /** Assigns \e array to the given array, replacing its current contents. */
      template< typename ArrayT >
      Array& operator = ( const ArrayT& array );

      /** \brief This function checks whether the given array is equal to \e array. */
      template< typename ArrayT >
      bool operator == ( const ArrayT& array ) const;

      /** \brief This function checks whether the given array is not equal to \e array. */
      template< typename ArrayT >
      bool operator != ( const ArrayT& array ) const;

      /**
       * \brief
       *
       * \param v Reference to a value.
       */
      void setValue( const Value& v );

      /**
       * \brief Checks if there is an element with value \e v in given array.
       *
       * \param v Reference to a value.
       */
      bool containsValue( const Value& v ) const;

      /**
       * \brief Checks if all elements in given array have the same value \e v.
       *
       * \param v Reference to a value.
       */
      bool containsOnlyValue( const Value& v ) const;

      /**
       * \brief Returns true if non-zero size is set.
       */
      operator bool() const;

      /**
       * \brief Method for saving the object to a \e file as a binary data.
       *
       * \param file Reference to a file.
       */
      bool save( File& file ) const;

      /**
       * Method for loading the object from a file as a binary data.
       *
       * \param file Reference to a file.
       */
      bool load( File& file );

      /**
       * \brief This method loads data without reallocation.
       *
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       */
      bool boundLoad( File& file );

      using Object::save;

      using Object::load;

      using Object::boundLoad;

      /** \brief Basic destructor. */
      ~Array();

   protected:

      /** \brief Method for releasing array data. */
      void releaseData() const;

      /** \brief Number of elements in array. */
      mutable Index size;

      /** \brief Pointer to data. */
      mutable Value* data;

      /**
       * \brief Pointer to the originally allocated data.
       *
       * They might differ if one long array is partitioned into more shorter
       * arrays. Each of them must know the pointer on allocated data because
       * the last one must deallocate the array. If outer data (not allocated
       * by TNL) are bind then this pointer is zero since no deallocation is
       * necessary.
       */
      mutable Value* allocationPointer;

      /**
       * \brief Counter of objects sharing this array or some parts of it.
       *
       * The reference counter is allocated after first sharing of the data
       * between more arrays. This is to avoid unnecessary dynamic memory allocation.
       */
      mutable int* referenceCounter;
};

template< typename Value, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Array< Value, Device, Index >& v );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Array_impl.h>
