/***************************************************************************
                          Array.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <list>
#include <vector>

#include <TNL/File.h>
#include <TNL/TypeTraits.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Containers/ArrayView.h>

namespace TNL {
/**
 * \brief Namespace for TNL containers.
 */
namespace Containers {

template< int, typename > class StaticArray;

/**
 * \brief \e Array is responsible for memory management, access to array
 * elements, and general array operations.
 *
 * \tparam Value  The type of array elements.
 * \tparam Device The device to be used for the execution of array operations.
 *                It can be either \ref Devices::Host or \ref Devices::Cuda.
 * \tparam Index  The indexing type.
 * \tparam Allocator The type of the allocator used for the allocation and
 *                   deallocation of memory used by the array. By default,
 *                   an appropriate allocator for the specified \e Device
 *                   is selected with \ref Allocators::Default.
 *
 * Memory management handled by constructors and destructors according to the
 * [RAII](https://en.wikipedia.org/wiki/RAII) principle and by methods
 * \ref setSize, \ref setLike, \ref swap, and \ref reset. You can also use
 * methods \ref getSize and \ref empty to check the current array size and
 * \ref getData to access the raw pointer.
 *
 * Methods annotated as \ref \_\_cuda_callable\_\_ can be called either from
 * host or from kernels executing on a device according to the \e Device
 * parameter. One of these methods is the \ref operator[] which provides direct
 * access to the array elements. However, it cannot be called from the host if
 * the array was allocated in a memory space which is not directly accessible
 * by the host. If the host needs access to individual array elements which are
 * allocated in a different memory space, they have to be accessed by the
 * \ref setElement or \ref getElement method. However, these methods imply an
 * explicit data transfer which is not buffered, so it can be very slow.
 *
 * Other methods, such as \ref operator=, \ref operator==, \ref operator!=,
 * \ref setValue, \ref containsValue, \ref containsOnlyValue, and \ref evaluate,
 * provide various operations on whole arrays.
 *
 * See also \ref ArrayView, \ref Vector, \ref VectorView.
 *
 * \par Example
 * \include ArrayExample.cpp
 * \par Output
 * \include ArrayExample.out
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class Array
{
   public:

      using ValueType = Value;
      using DeviceType = Device;
      using IndexType = Index;
      using AllocatorType = Allocator;
      using HostType = Containers::Array< Value, Devices::Host, Index >;
      using CudaType = Containers::Array< Value, Devices::Cuda, Index >;
      using ViewType = ArrayView< Value, Device, Index >;
      using ConstViewType = ArrayView< std::add_const_t< Value >, Device, Index >;

      /**
       * \brief Constructs an empty array with zero size.
       */
      Array() = default;

      /**
       * \brief Constructs an empty array and sets the provided allocator.
       *
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( const AllocatorType& allocator );

      /**
       * \brief Constructs an array with given size.
       *
       * \param size The number of array elements to be allocated.
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( const IndexType& size, const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Constructs an array with given size and copies data from given
       * pointer.
       *
       * \param data The pointer to the data to be copied to the array.
       * \param size The number of array elements to be copied to the array.
       * \param allocator The allocator to be associated with this array.
       */
      Array( Value* data,
             const IndexType& size,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Copy constructor.
       *
       * \param array The array to be copied.
       */
      explicit Array( const Array& array );

      /**
       * \brief Copy constructor with a specific allocator.
       *
       * \param array The array to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( const Array& array, const AllocatorType& allocator );

      /**
       * \brief Copy constructor.
       *
       * \param array The array to be copied.
       * \param begin The first index which should be copied.
       * \param size The number of elements that should be copied.
       * \param allocator The allocator to be associated with this array.
       */
      Array( const Array& array,
             IndexType begin,
             IndexType size = 0,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Move constructor for initialization from \e rvalues.
       *
       * \param array The array to be moved.
       */
      Array( Array&& array ) = default;

      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::initializer_list, e.g. `{...}`.
       *
       * \param list The initializer list containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::initializer_list< InValue >& list,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::list.
       *
       * \param list The STL list containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::list< InValue >& list,
             const AllocatorType& allocator = AllocatorType() );


      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::vector.
       *
       * \param vector The STL vector containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::vector< InValue >& vector,
             const AllocatorType& allocator = AllocatorType() );


      /**
       * \brief Returns the allocator associated with the array.
       */
      AllocatorType getAllocator() const;

      /**
       * \brief Returns a \ref String representation of the array type in C++ style.
       */
      static String getType();

      /**
       * \brief Returns a \ref String representation of the array type in C++ style.
       */
      virtual String getTypeVirtual() const;

      /**
       * \brief Returns a \ref String representation of the array type in C++ style,
       * where device is always \ref Devices::Host.
       */
      static String getSerializationType();

      /**
       * \brief Returns a \ref String representation of the array type in C++ style,
       * where device is always \ref Devices::Host.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for setting the array size.
       *
       * If the array shares data with other arrays, the data is unbound. If the
       * current data is not shared and the current size is the same as the new
       * one, nothing happens.
       *
       * If the array size changes, the current data will be deallocated, thus
       * all pointers and views to the array alements will become invalid.
       *
       * \param size The new size of the array.
       */
      void setSize( Index size );

      /**
       * \brief Returns the current array size.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ Index getSize() const;

      /**
       * \brief Sets the same size as the size of an existing array.
       *
       * If the array size changes, the current data will be deallocated, thus
       * all pointers and views to the array alements will become invalid.
       *
       * \tparam ArrayT The type of the parameter can be any type which provides
       *         the method \ref getSize() with the same signature as \e Array.
       * \param array The array whose size is to be taken.
       */
      template< typename ArrayT >
      void setLike( const ArrayT& array );

      /**
       * \brief Binds \e _data with this array.
       *
       * Releases old data and binds this array with new \e _data. Also sets new
       * \e _size of this array.
       *
       * This method is deprecated, use \ref ArrayView instead.
       *
       * \param _data Pointer to new data.
       * \param _size Size of new _data. Number of elements.
       */
      [[deprecated("Binding functionality of Array is deprecated, ArrayView should be used instead.")]]
      void bind( Value* _data,
                 const Index _size );

      /**
       * \brief Binds this array with another \e array.
       *
       * Releases old data and binds this array with new \e array starting at
       * position \e begin. Also sets new \e size of this array.
       *
       * This method is deprecated, use \ref ArrayView instead.
       *
       * \tparam ArrayT Type of array.
       * \param array Reference to a new array.
       * \param begin Starting index position.
       * \param size Size of new array. Number of elements.
       */
      template< typename ArrayT >
      [[deprecated("Binding functionality of Array is deprecated, ArrayView should be used instead.")]]
      void bind( const ArrayT& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      /**
       * \brief Binds this array with a static array of size \e Size.
       *
       * Releases old data and binds this array with a static array of size \e
       * Size.
       *
       * This method is deprecated, use \ref ArrayView instead.
       *
       * \tparam Size Size of array.
       * \param array Reference to a static array.
       */
      template< int Size >
      [[deprecated("Binding functionality of Array is deprecated, ArrayView should be used instead.")]]
      void bind( StaticArray< Size, Value >& array );

      /**
       * \brief Returns a modifiable view of the array.
       *
       * By default, a view for the whole array is returned. If \e begin or
       * \e end is set to a non-zero value, a view only for the sub-interval
       * `[begin, end)` is returned.
       *
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      ViewType getView( IndexType begin = 0, IndexType end = 0 );

      /**
       * \brief Returns a non-modifiable view of the array.
       *
       * By default, a view for the whole array is returned. If \e begin or
       * \e end is set to a non-zero value, a view only for the sub-interval
       * `[begin, end)` is returned.
       *
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      ConstViewType getView( IndexType begin = 0, IndexType end = 0 ) const;

      /**
       * \brief Returns a non-modifiable view of the array.
       *
       * By default, a view for the whole array is returned. If \e begin or
       * \e end is set to a non-zero value, a view only for the sub-interval
       * `[begin, end)` is returned.
       *
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

      /**
       * \brief Conversion operator to a modifiable view of the array.
       */
      operator ViewType();

      /**
       * \brief Conversion operator to a non-modifiable view of the array.
       */
      operator ConstViewType() const;

      /**
       * \brief Swaps this array with another.
       *
       * Swapping is done in a shallow way, i.e. only pointers and sizes are
       * swapped.
       *
       * \param array The array to be swapped with this array.
       */
      void swap( Array& array );

      /**
       * \brief Resets the array to the empty state.
       *
       * The current data will be deallocated, thus all pointers and views to
       * the array alements will become invalid.
       */
      void reset();

      /**
       * \brief Returns a \e const-qualified raw pointer to the data.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ const Value* getData() const;

      /**
       * \brief Returns a raw pointer to the data.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ Value* getData();

      /**
       * \brief Returns a \e const-qualified raw pointer to the data.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ const Value* getArrayData() const;

      /**
       * \brief Returns a raw pointer to the data.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ Value* getArrayData();


      /**
       * \brief Sets the value of the \e i-th element to \e v.
       *
       * This method can be called only from the host, but even for arrays
       * allocated in a different memory space (e.g. GPU global memory).
       *
       * \param i The index of the element to be set.
       * \param v The new value of the element.
       */
      void setElement( const Index& i, const Value& v );

      /**
       * \brief Returns the value of the \e i-th element.
       *
       * This method can be called only from the host, but even for arrays
       * allocated in a different memory space (e.g. GPU global memory).
       *
       * \param i The index of the element to be returned.
       */
      Value getElement( const Index& i ) const;

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * This method can be called only from the device which has direct access
       * to the memory space where the array was allocated. For example, if the
       * array was allocated in the host memory, it can be called only from
       * host, and if the array was allocated in the device memory, it can be
       * called only from device kernels.
       *
       * \param i The index of the element to be accessed.
       * \return Reference to the \e i-th element.
       */
      __cuda_callable__ inline Value& operator[]( const Index& i );

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * This method can be called only from the device which has direct access
       * to the memory space where the array was allocated. For example, if the
       * array was allocated in the host memory, it can be called only from
       * host, and if the array was allocated in the device memory, it can be
       * called only from device kernels.
       *
       * \param i The index of the element to be accessed.
       * \return Constant reference to the \e i-th element.
       */
      __cuda_callable__ inline const Value& operator[]( const Index& i ) const;

      /**
       * \brief Copy-assignment operator for copying data from another array.
       *
       * \param array Reference to the source array.
       * \return Reference to this array.
       */
      Array& operator=( const Array& array );

      /**
       * \brief Move-assignment operator for acquiring data from \e rvalues.
       *
       * \param array Reference to the source array.
       * \return Reference to this array.
       */
      Array& operator=( Array&& array );

      /**
       * \brief Assigns either array-like container or a single value.
       *
       * If \e T is an array type, e.g. \ref Array, \ref ArrayView,
       * \ref StaticArray, \ref Vector, \ref VectorView, or \ref StaticVector,
       * the elements from \e data are copied into this array. Otherwise, if it
       * is a type convertible to \ref ValueType, all array elements are set to
       * the value \e data.
       *
       * \tparam T The type of the source array or value.
       * \param data Reference to the source array or value.
       * \return Reference to this array.
       */
      template< typename T >
      Array& operator=( const T& data );

      /**
       * \brief Copies elements from \ref std::list to this array.
       *
       * \param list The STL list containing elements to be copied.
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator=( const std::list< InValue >& list );

      /**
       * \brief Copies elements from \ref std::vector to this array.
       *
       * \param list The STL vector containing elements to be copied.
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator=( const std::vector< InValue >& vector );

      /**
       * \brief Compares the array with another array-like container.
       *
       * \tparam ArrayT The type of the parameter can be any array-like
       *         container, e.g. \ref Array, \ref ArrayView, \ref Vector,
       *         \ref VectorView, etc.
       * \param array Reference to the array-like container.
       * \return \ref True if both arrays are element-wise equal and \ref false
       *         otherwise.
       */
      template< typename ArrayT >
      bool operator==( const ArrayT& array ) const;

      /**
       * \brief Compares the array with another array-like container.
       *
       * \tparam ArrayT The type of the parameter can be any array-like
       *         container, e.g. \ref Array, \ref ArrayView, \ref Vector,
       *         \ref VectorView, etc.
       * \param array Reference to the array-like container.
       * \return The negated result of \ref operator==.
       */
      template< typename ArrayT >
      bool operator!=( const ArrayT& array ) const;

      /**
       * \brief Sets elements of the array to given value.
       *
       * By default, all array elements are set to the given value. If \e begin
       * or \e end is set to a non-zero value, only elements in the sub-interval
       * `[begin, end)` are set.
       *
       * \param v The new value for the array elements.
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      void setValue( const ValueType& v,
                     IndexType begin = 0,
                     IndexType end = 0 );

      /**
       * \brief Sets the array elements using given lambda function.
       *
       * Evaluates a lambda function \e f on whole array or just on its
       * sub-interval `[begin, end)`. This is performed at the same place
       * where the array is allocated, i.e. it is efficient even on GPU.
       *
       * \param f The lambda function to be evaluated.
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      template< typename Function >
      void evaluate( const Function& f,
                     IndexType begin = 0,
                     IndexType end = 0 );

      /**
       * \brief Checks if there is an element with value \e v.
       *
       * By default, all elements of the array are checked. If \e begin or
       * \e end is set to a non-zero value, only elements in the sub-interval
       * `[begin, end)` are checked.
       *
       * \param v The value to be checked.
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       * \return True if there is _at least one_ element in the sub-interval
       *         `[begin, end)` which has the value \e v.
       */
      bool containsValue( const ValueType& v,
                          IndexType begin = 0,
                          IndexType end = 0 ) const;

      /**
       * \brief Checks if all elements have the same value \e v.
       *
       * By default, all elements of the array are checked. If \e begin or
       * \e end is set to a non-zero value, only elements in the sub-interval
       * `[begin, end)` are checked.
       *
       * \param v The value to be checked.
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       * \return True if there is _all_ elements in the sub-interval
       *         `[begin, end)` have the same value \e v.
       */
      bool containsOnlyValue( const ValueType& v,
                              IndexType begin = 0,
                              IndexType end = 0 ) const;

      /**
       * \brief Returns \e true if the current array size is zero.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__
      bool empty() const;

      /**
       * \brief Method for saving the array to a binary file \e fileName.
       *
       * \param fileName The output file name.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for loading the array from a binary file \e fileName.
       *
       * \param fileName The input file name.
       */
      void load( const String& fileName );

      /** \brief Destructor. */
      ~Array();

   protected:

      /** \brief Method for releasing (deallocating) array data. */
      void releaseData();

      /** \brief Number of elements in the array. */
      mutable Index size = 0;

      /** \brief Pointer to the data. */
      mutable Value* data = nullptr;

      /**
       * \brief Pointer to the originally allocated data.
       *
       * They might differ if one long array is partitioned into more shorter
       * arrays. Each of them must know the pointer on allocated data because
       * the last one must deallocate the array. If outer data (not allocated
       * by TNL) are bind then this pointer is zero since no deallocation is
       * necessary.
       */
      mutable Value* allocationPointer = nullptr;

      /**
       * \brief Counter of objects sharing this array or some parts of it.
       *
       * The reference counter is allocated after first sharing of the data between
       * more arrays. This is to avoid unnecessary dynamic memory allocation.
       */
      mutable int* referenceCounter = nullptr;

      /**
       * \brief The internal allocator instance.
       */
      Allocator allocator;
};

template< typename Value, typename Device, typename Index, typename Allocator >
std::ostream& operator<<( std::ostream& str, const Array< Value, Device, Index, Allocator >& array );

/**
 * \brief Serialization of arrays into binary files.
 */
template< typename Value, typename Device, typename Index, typename Allocator >
File& operator<<( File& file, const Array< Value, Device, Index, Allocator >& array );

template< typename Value, typename Device, typename Index, typename Allocator >
File& operator<<( File&& file, const Array< Value, Device, Index, Allocator >& array );

/**
 * \brief Deserialization of arrays from binary files.
 */
template< typename Value, typename Device, typename Index, typename Allocator >
File& operator>>( File& file, Array< Value, Device, Index, Allocator >& array );

template< typename Value, typename Device, typename Index, typename Allocator >
File& operator>>( File&& file, Array< Value, Device, Index, Allocator >& array );

} // namespace Containers

template< typename Value_, typename Device, typename Index >
struct IsStatic< Containers::Array< Value_, Device, Index > >
{
   static constexpr bool Value = false;
};

} // namespace TNL

#include <TNL/Containers/Array.hpp>
