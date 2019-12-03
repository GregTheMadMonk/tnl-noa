/***************************************************************************
                          ArrayView.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>  // std::add_const_t

#include <TNL/TypeTraits.h>
#include <TNL/File.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {

/**
 * \brief \e ArrayView is a simple data structure which provides a non-owning
 * encapsulation of array data. That is, \e ArrayView is like \ref Array without
 * memory management.
 *
 * The meaning of the template parameters is the same as in \ref Array.
 *
 * \tparam Value  The type of array elements.
 * \tparam Device The device where the array is allocated. This ensures the
 *                compile-time checks of correct pointers manipulation. It can
 *                be either \ref Devices::Host or \ref Devices::Cuda.
 * \tparam Index  The indexing type.
 *
 * \e ArrayView provides access to array elements and general array operations
 * same as \ref Array, but it does not manage memory. The construction of an
 * \e ArrayView does not make a memory allocation, but it can be _bound_ to an
 * existing portion of memory specified by a raw pointer, another \e ArrayView,
 * or an \e Array. Similarly, \e ArrayView is not resizable, so there is no
 * \e setSize method. Note that \e ArrayView does not _own_ its data, so it does
 * not deallocate it in the destructor.
 *
 * Another important difference between \e ArrayView and \ref Array is in the
 * copy semantics. While the copy-constructor of \e Array makes a deep copy of
 * the data, \e ArrayView makes only a shallow copy in the copy-constructor,
 * i.e. it changes only its pointer and size. As a result, array views can be
 * efficiently _passed by value_ (even to device kernels) or _captured by value_
 * in lambda functions (even in device lambda functions). Note that
 * \ref operator= in both \e ArrayView and \e Array still makes a deep copy of
 * the data.
 *
 * See also \ref Array, \ref Vector, \ref VectorView.
 *
 * \par Example
 * \include Containers/ArrayViewExample.cpp
 * \par Output
 * \include ArrayViewExample.out
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
class ArrayView
{
public:
   /**
    * \brief Type of elements stored in this array.
    */
   using ValueType = Value;

   /**
    * \brief Device where the array is allocated.
    * 
    * See \ref Devices::Host or \ref Devices::Cuda.
    */
   using DeviceType = Device;

   /**
    * \brief Type being used for the array elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Compatible ArrayView type.
    */
   using ViewType = ArrayView< Value, Device, Index >;

   /**
    * \brief Compatible constant ArrayView type.
    */
   using ConstViewType = ArrayView< std::add_const_t< Value >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain an \ref ArrayView type with changed template parameters.
    */
   template< typename _Value,
             typename _Device = Device,
             typename _Index = Index >
   using Self = ArrayView< _Value, _Device, _Index >;


   /**
    * \brief Constructs an empty array view.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   ArrayView() = default;

   /**
    * \brief Constructs an array view by binding to the given data pointer
    * and size.
    *
    * This method can be called from device kernels.
    *
    * \param data The data pointer to be bound.
    * \param size The number of elements in the array view.
    */
   __cuda_callable__
   ArrayView( Value* data, Index size );

   /**
    * \brief Shallow copy constructor.
    *
    * This method can be called from device kernels.
    *
    * \param view The array view to be copied.
    */
   __cuda_callable__
   ArrayView( const ArrayView& view ) = default;

   /**
    * \brief "Templated" shallow copy constructor.
    *
    * This method can be called from device kernels.
    *
    * \tparam Value_ The template parameter can be any cv-qualified variant of
    *                \e ValueType.
    * \param view The array view to be copied.
    */
   template< typename Value_ >
   __cuda_callable__
   ArrayView( const ArrayView< Value_, Device, Index >& view )
   : data(view.getData()), size(view.getSize()) {}

   /**
    * \brief Move constructor for initialization from \e rvalues.
    *
    * This method can be called from device kernels.
    *
    * \param view The array view to be moved.
    */
   __cuda_callable__
   ArrayView( ArrayView&& view ) = default;

   /**
    * \brief Method for rebinding (reinitialization) using a raw pointer and
    * size.
    *
    * This method can be called from device kernels.
    *
    * \param data The data pointer to be bound to the array view.
    * \param size The number of elements in the array view.
    */
   __cuda_callable__
   void bind( Value* data, const Index size );

   /**
    * \brief Method for rebinding (reinitialization) using another array view.
    *
    * Note that you can also bind directly to an \e Array instance and other
    * objects whose type is implicitly convertible to \e ArrayView.
    *
    * This method can be called from device kernels.
    *
    * \param view The array view to be bound.
    */
   __cuda_callable__
   void bind( ArrayView view );

   /**
    * \brief Returns a modifiable view of the array view.
    *
    * By default, a view for the whole array is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ViewType getView( const IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the array view.
    *
    * By default, a view for the whole array is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ConstViewType getConstView( const IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Deep copy assignment operator for copying data from another array
    * view.
    *
    * \param view Reference to the source array view.
    * \return Reference to this array view.
    */
   ArrayView& operator=( const ArrayView& view );

   /**
    * \brief Assigns either array-like container or a single value.
    *
    * If \e T is an array type, e.g. \ref Array, \ref ArrayView,
    * \ref StaticArray, \ref Vector, \ref VectorView, or \ref StaticVector,
    * the elements from \e data are copied into this array view. Otherwise, if
    * it is a type convertible to \ref ValueType, all array elements are set to
    * the value \e data.
    *
    * \tparam T The type of the source array or value.
    * \param data Reference to the source array or value.
    * \return Reference to this array view.
    */
   template< typename T,
             typename...,
             typename = std::enable_if_t< std::is_convertible< T, ValueType >::value || IsArrayType< T >::value > >
   ArrayView& operator=( const T& array );

   /**
    * \brief Swaps this array view with another.
    *
    * Swapping is done in a shallow way, i.e. only pointers and sizes are
    * swapped.
    *
    * \param view The array view to be swapped with this array view.
    */
   __cuda_callable__
   void swap( ArrayView& view );

   /***
    * \brief Resets the array view to the empty state.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   void reset();

   /**
    * \brief Returns \e true if the current array view size is zero.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   bool empty() const;

   /**
    * \brief Returns a \e const-qualified raw pointer to the data.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   const Value* getData() const;

   /**
    * \brief Returns a raw pointer to the data.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   Value* getData();

   /**
    * \brief Returns a \e const-qualified raw pointer to the data.
    *
    * Use this method in algorithms where you want to emphasize that
    * C-style array pointer is required.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   const Value* getArrayData() const;

   /**
    * \brief Returns a raw pointer to the data.
    *
    * Use this method in algorithms where you want to emphasize that
    * C-style array pointer is required.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   Value* getArrayData();

   /**
    * \brief Returns the current size of the array view.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   Index getSize() const;

   /**
    * \brief Sets the value of the \e i-th element to \e v.
    *
    * This method can be called only from the host, but even for array views
    * allocated in a different memory space (e.g. GPU global memory).
    *
    * \param i The index of the element to be set.
    * \param v The new value of the element.
    */
   void setElement( Index i, Value value );

   /**
    * \brief Returns the value of the \e i-th element.
    *
    * This method can be called only from the host, but even for array views
    * allocated in a different memory space (e.g. GPU global memory).
    *
    * \param i The index of the element to be returned.
    */
   Value getElement( Index i ) const;

   /**
    * \brief Accesses the \e i-th element of the array view.
    *
    * This method can be called only from the device which has direct access
    * to the memory space where the data was allocated. For example, if the
    * data was allocated in the host memory, it can be called only from
    * host, and if the data was allocated in the device memory, it can be
    * called only from device kernels.
    *
    * \param i The index of the element to be accessed.
    * \return Reference to the \e i-th element.
    */
   __cuda_callable__
   Value& operator[]( Index i );

   /**
    * \brief Accesses the \e i-th element of the array view.
    *
    * This method can be called only from the device which has direct access
    * to the memory space where the data was allocated. For example, if the
    * data was allocated in the host memory, it can be called only from
    * host, and if the data was allocated in the device memory, it can be
    * called only from device kernels.
    *
    * \param i The index of the element to be accessed.
    * \return Constant reference to the \e i-th element.
    */
   __cuda_callable__
   const Value& operator[]( Index i ) const;

   /**
    * \brief Compares the array view with another array-like container.
    *
    * \tparam ArrayT The type of the parameter can be any array-like
    *         container, e.g. \ref Array, \ref ArrayView, \ref Vector,
    *         \ref VectorView, etc.
    * \param array Reference to the array-like container.
    * \return \ref True if the array view is element-wise equal to the
    *         array-like container and \ref false otherwise.
    */
   template< typename ArrayT >
   bool operator==( const ArrayT& array ) const;

   /**
    * \brief Compares the array view with another array-like container.
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
    * \brief Sets elements of the array view to given value.
    *
    * By default, all array view elements are set to the given value. If
    * \e begin or \e end is set to a non-zero value, only elements in the
    * sub-interval `[begin, end)` are set.
    *
    * \param v The new value for the array view elements.
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array view size.
    */
   void setValue( Value value,
                  const Index begin = 0,
                  Index end = 0 );

   /**
    * \brief Sets the array view elements using given lambda function.
    *
    * Evaluates a lambda function \e f on whole array view or just on its
    * sub-interval `[begin, end)`. This is performed at the same place
    * where the data is allocated, i.e. it is efficient even on GPU.
    *
    * \param f The lambda function to be evaluated.
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array view size.
    */
   template< typename Function >
   void evaluate( const Function& f,
                  const Index begin = 0,
                  Index end = 0 );

   /**
    * \brief Checks if there is an element with value \e v.
    *
    * By default, all elements of the array view are checked. If \e begin or
    * \e end is set to a non-zero value, only elements in the sub-interval
    * `[begin, end)` are checked.
    *
    * \param v The value to be checked.
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array view size.
    * \return True if there is _at least one_ element in the sub-interval
    *         `[begin, end)` which has the value \e v.
    */
   bool containsValue( Value value,
                       const Index begin = 0,
                       Index end = 0  ) const;

   /**
    * \brief Checks if all elements have the same value \e v.
    *
    * By default, all elements of the array view are checked. If \e begin or
    * \e end is set to a non-zero value, only elements in the sub-interval
    * `[begin, end)` are checked.
    *
    * \param v The value to be checked.
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array view size.
    * \return True if there is _all_ elements in the sub-interval
    *         `[begin, end)` have the same value \e v.
    */
   bool containsOnlyValue( Value value,
                           const Index begin = 0,
                           Index end = 0  ) const;

   /**
    * \brief Method for saving the data to a binary file \e fileName.
    *
    * \param fileName The output file name.
    */
   void save( const String& fileName ) const;

   /**
    * \brief Method for loading the data from a binary file \e fileName.
    *
    * \param fileName The input file name.
    */
   void load( const String& fileName );

protected:
   //! Pointer to the data
   Value* data = nullptr;

   //! Array view size
   Index size = 0;
};

template< typename Value, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Value, Device, Index >& view );

/**
 * \brief Serialization of array views into binary files.
 */
template< typename Value, typename Device, typename Index >
File& operator<<( File& file, const ArrayView< Value, Device, Index > view );

template< typename Value, typename Device, typename Index >
File& operator<<( File&& file, const ArrayView< Value, Device, Index > view );

/**
 * \brief Deserialization of array views from binary files.
 */
template< typename Value, typename Device, typename Index >
File& operator>>( File& file, ArrayView< Value, Device, Index > view );

template< typename Value, typename Device, typename Index >
File& operator>>( File&& file, ArrayView< Value, Device, Index > view );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/ArrayView.hpp>
