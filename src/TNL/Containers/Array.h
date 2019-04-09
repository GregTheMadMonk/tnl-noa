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
#include <TNL/Object.h>
#include <TNL/File.h>
#include <TNL/TypeTraits.h>
#include <TNL/Devices/Host.h>

namespace TNL {
/**
 * \brief Namespace for TNL containers.
 */
namespace Containers {

template< int, typename > class StaticArray;

/**
 * \brief Array is responsible for memory management, basic elements
 * manipulation and I/O operations.
 *
 * \tparam Value is type of array elements.
 * \tparam Device is device where the array is going to be allocated - some of \ref Devices::Host and \ref Devices::Cuda.
 * \tparam Index is indexing type.
 *
 * In the \e Device type, the Array remembers where the memory is allocated.
 * This ensures the compile-time checks of correct pointers manipulation.
 * Methods defined as \ref __cuda_callable__ can be called even from kernels
 * running on device. Array elements can be changed either using the \ref operator[]
 * which is more efficient but it can be called from CPU only for arrays
 * allocated on host (CPU) and when the array is allocated on GPU, the operator[]
 * can be called only from kernels running on the device (GPU). On the other
 * hand, methods \ref setElement and \ref getElement, can be called only from the
 * host (CPU) does not matter if the array is allocated on the host or the device.
 * In the latter case, explicit data transfer between host and device (via PCI
 * express or NVlink in more lucky systems) is invoked and so it can be very
 * slow. In not time critical parts of code, this is not an issue, however.
 * Another way to change data stored in the array is \ref evaluate which evaluates
 * given lambda function. This is performed at the same place where the array is
 * allocated i.e. it is efficient even on GPU. For simple checking of the array
 * contents, one may use methods \ref containValue and \ref containsValue and
 * \ref containsOnlyValue.
 * Array also offers data sharing using methods \ref bind. This is, however, obsolete
 * and will be soon replaced with proxy object \ref ArrayView.
 *
 * \par Example
 * \include ArrayExample.cpp
 *
 * See also \ref Containers::ArravView, \ref Containers::Vector, \ref Containers::VectorView.
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
class Array : public Object
{
   public:

      using ValueType = Value;
      using DeviceType = Device;
      using IndexType = Index;
      using ThisType = Containers::Array< ValueType, DeviceType, IndexType >;
      using HostType = Containers::Array< Value, Devices::Host, Index >;
      using CudaType = Containers::Array< Value, Devices::Cuda, Index >;

      /**
       * \brief Basic constructor.
       *
       * Constructs an empty array with zero size.
       */
      Array();

      /**
       * \brief Constructor with array size.
       *
       * \param size is number of array elements.
       */
      Array( const IndexType& size );

      /**
       * \brief Constructor with data pointer and size.
       *
       * In this case, the Array just encapsulates the pointer \e data. No
       * deallocation is done in destructor.
       *
       * This behavior of the Array is obsolete and \ref ArrayView should be used
       * instead.
       *
       * \param data Pointer to data.
       * \param size Number of array elements.
       */
      Array( Value* data,
             const IndexType& size );

      /**
       * \brief Copy constructor.
       *
       * \param array is an array to be copied.
       */
      explicit Array( const Array& array );

      /**
       * \brief Bind constructor .
       *
       * The constructor does not make a deep copy, but binds to the supplied array.
       * This is also obsolete, \ref ArraView should be used instead.
       *
       * \param array is an array that is to be bound.
       * \param begin is the first index which should be bound.
       * \param size is number of array elements that should be bound.
       */
      Array( Array& array,
             const IndexType& begin = 0,
             const IndexType& size = 0 );

      /**
       * \brief Move constructor.
       *
       * @param array is an array to be moved
       */
      Array( Array&& array );

      /**
       * \brief Initialize the array from initializer list, i.e. { ... }
       *
       * @param list Initializer list.
       */
      template< typename InValue >
      Array( const std::initializer_list< InValue >& list );

      /**
       * \brief Initialize the array from std::list.
       *
       * @param list Input STL list.
       */
      template< typename InValue >
      Array( const std::list< InValue >& list );

      /**
       * \brief Initialize the array from std::vector.
       *
       * @param vector Input STL vector.
       */
      template< typename InValue >
      Array( const std::vector< InValue >& vector );

      /**
       * \brief Returns type of array in C++ style.
       *
       * \return String with array type.
       */
      static String getType();

      /**
       * \brief Returns type of array in C++ style.
       *
       * \return String with array type.
       */
      virtual String getTypeVirtual() const;

      /**
       *  \brief Returns type of array in C++ style where device is always \ref Devices::Host.
       *
       * \return String with serialization array type.
       */
      static String getSerializationType();

      /**
       *  \brief Returns type of array in C++ style where device is always \ref Devices::Host.
       *
       * \return String with serialization array type.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for setting the array size.
       *
       * If the array shares data with other arrays these data are released.
       * If the current data are not shared and the current size is the same
       * as the new one, nothing happens.
       *
       * \param size is number of array elements.
       */
      void setSize( Index size );

      /**
       * \brief Method for getting the size of an array.
       *
       * This method can be called from device kernels.
       *
       * \return Array size.
       */
      __cuda_callable__ Index getSize() const;

      /**
       * \brief Assigns features of the existing \e array to the given array.
       *
       * Sets the same size as the size of existing \e array.
       *
       * \tparam ArrayT is any array type having method \ref getSize().
       * \param array is reference to the source array.
       */
      template< typename ArrayT >
      void setLike( const ArrayT& array );

      /**
       * \brief Binds \e _data with this array.
       *
       * Releases old data and binds this array with new \e _data. Also sets new
       * \e _size of this array.
       *
       * This method is obsolete, use \ref ArrayView instead.
       *
       * \param _data Pointer to new data.
       * \param _size Size of new _data. Number of elements.
       */
      void bind( Value* _data,
                 const Index _size );

      /**
       * \brief Binds this array with another \e array.
       *
       * Releases old data and binds this array with new \e array starting at
       * position \e begin. Also sets new \e size of this array.
       *
       * This method is obsolete, use \ref ArrayView instead.
       *
       * \tparam ArrayT Type of array.
       * \param array Reference to a new array.
       * \param begin Starting index position.
       * \param size Size of new array. Number of elements.
       */
      template< typename ArrayT >
      void bind( const ArrayT& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      /**
       * \brief Binds this array with a static array of size \e Size.
       *
       * Releases old data and binds this array with a static array of size \e
       * Size.
       *
       * This method is obsolete, use \ref ArrayView instead.
       *
       * \tparam Size Size of array.
       * \param array Reference to a static array.
       */
      template< int Size >
      void bind( StaticArray< Size, Value >& array );

      /**
       * \brief Swaps this array with another.
       *
       * The swap is done in a shallow way, i.e. swapping only pointers and sizes.
       *
       * \param array is the array to be swapped with this array.
       */
      void swap( Array& array );

      /**
       * \brief Resets the array.
       *
       * Releases the array to empty state.
       */
      void reset();

      /**
       * \brief Data pointer getter for constant instances.
       *
       * This method can be called from device kernels.
       *
       * \return Pointer to array data.
       */
      __cuda_callable__ const Value* getData() const;

      /**
       * \brief Data pointer getter.
       *
       * This method can be called from device kernels.
       *
       * \return Pointer to array data.
       */
      __cuda_callable__ Value* getData();

      /**
       * \brief Data pointer getter for constant instances.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       *
       * \return Pointer to array data.
       */
      __cuda_callable__ const Value* getArrayData() const;

      /**
       * \brief Data pointer getter.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       *
       * \return Pointer to array data.
       */
      __cuda_callable__ Value* getArrayData();


      /**
       * \brief Array elements setter - change value of an element at position \e i.
       *
       * This method can be called only from the host system (CPU) but even for
       * arrays allocated on device (GPU).
       *
       * \param i is element index.
       * \param v is the new value of the element.
       */
      void setElement( const Index& i, const Value& v );

      /**
       * \brief Array elements getter - returns value of an element at position \e i.
       *
       * This method can be called only from the host system (CPU) but even for
       * arrays allocated on device (GPU).
       *
       * \param i Index position of an element.
       *
       * \return Copy of i-th element.
       */
      Value getElement( const Index& i ) const;

      /**
       * \brief Accesses specified element at the position \e i.
       *
       * This method can be called from device (GPU) kernels if the array is allocated
       * on the device. In this case, it cannot be called from host (CPU.)
       *
       * \param i is position of the element.
       *
       * \return Reference to i-th element.
       */
      __cuda_callable__ inline Value& operator[] ( const Index& i );

      /**
       * \brief Accesses specified element at the position \e i.
       *
       * This method can be called from device (GPU) kernels if the array is allocated
       * on the device. In this case, it cannot be called from host (CPU.)
       *
       * \param i is position of the element.
       *
       * \return Constant reference to i-th pointer.
       */
      __cuda_callable__ inline const Value& operator[] ( const Index& i ) const;

      /**
       * \brief Assigns \e array to this array, replacing its current contents.
       *
       * \param array is reference to the array.
       *
       * \return Reference to this array.
       */
      Array& operator = ( const Array& array );

      /**
       * \brief Move contents of \e array to this array.
       *
       * \param array is reference to the array.
       *
       * \return Reference to this array.
       */
      Array& operator = ( Array&& array );

      /**
       * \brief Assigns either array-like container or single value.
       *
       * If \e T is array type i.e. \ref Array, \ref ArrayView, \ref StaticArray,
       * \ref Vector, \ref VectorView, \ref StaticVector, \ref DistributedArray,
       * \ref DistributedArrayView, \ref DistributedVector or
       * \ref DistributedVectorView, its elements are copied into this array. If
       * it is other type convertibly to Array::ValueType, all array elements are
       * set to the value \e data.
       *
       * \tparam T is type of array or value type.
       *
       * \param data is a reference to array or value.
       *
       * \return Reference to this array.
       */
      template< typename T >
      Array& operator = ( const T& data );

      /**
       * \brief Assigns STL list to this array.
       *
       * \param list is STL list
       *
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator = ( const std::list< InValue >& list );

      /**
       * \brief Assigns STL vector to this array.
       *
       * \param vector is STL vector
       *
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator = ( const std::vector< InValue >& vector );

      /**
       * \brief Comparison operator with another array-like container.
       *
       * \tparam ArrayT is type of an array-like container, i.e Array, ArrayView, Vector, VectorView, DistributedArray, DistributedVector etc.
       * \param array is reference to an array.
       *
       * \return True if both arrays are equal element-wise and false otherwise.
       */
      template< typename ArrayT >
      bool operator == ( const ArrayT& array ) const;

      /**
       * \brief This function checks whether this array is not equal to \e array.
       *
       * \tparam ArrayT Type of array.
       * \param array Reference to an array.
       *
       * \return True if both arrays are not equal element-wise and false otherwise.
       */
      template< typename ArrayT >
      bool operator != ( const ArrayT& array ) const;

      /**
       * \brief Sets the array elements to given value.
       *
       * Sets all the array values to \e v.
       *
       * \param v Reference to a value.
       */
      void setValue( const Value& v,
                     const Index begin = 0,
                     Index end = -1 );

      /**
       * \brief Checks if there is an element with value \e v.
       *
       * By default, the method checks all array elements. By setting indexes
       * \e begin and \e end, only elements in given interval are checked.
       *
       * \param v is reference to the value.
       * \param begin is the first element to be checked
       * \param end is the last element to be checked. If \e end equals -1, its
       * value is replaces by the array size.
       *
       * \return True if there is **at least one** array element in interval [\e begin, \e end ) having value \e v.
       */
      bool containsValue( const Value& v,
                          const Index begin = 0,
                          Index end = -1 ) const;

      /**
       * \brief Checks if all elements have the same value \e v.
       *
       * By default, the method checks all array elements. By setting indexes
       * \e begin and \e end, only elements in given interval are checked.
       *
       * \param v Reference to a value.
       * \param begin is the first element to be checked
       * \param end is the last element to be checked. If \e end equals -1, its
       * value is replaces by the array size.
       *
       * \return True if there **all** array elements in interval [\e begin, \e end ) have value \e v.
       */
      bool containsOnlyValue( const Value& v,
                              const Index begin = 0,
                              Index end = -1 ) const;

      /**
       * \brief Returns true if non-zero size is set.
       *
       * This method can be called from device kernels.
       *
       * \return Returns \e true if array view size is zero, \e false otherwise.
       */
      __cuda_callable__
      bool empty() const;

      /**
       * \brief Method for saving the object to a \e file as a binary data.
       *
       * \param file Reference to a file.
       */
      void save( File& file ) const;

      /**
       * Method for loading the object from a file as a binary data.
       *
       * \param file Reference to a file.
       */
      void load( File& file );

      /**
       * \brief This method loads data without reallocation.
       *
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       *
       * This method is deprecated - use ArrayView instead.
       */
      void boundLoad( File& file );

      using Object::save;

      using Object::load;

      using Object::boundLoad;

      /** \brief Basic destructor. */
      ~Array();

   protected:

      /** \brief Method for releasing array data. */
      void releaseData() const;

      /** \brief Number of elements in array. */
      mutable Index size = 0;

      /** \brief Pointer to data. */
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
};

template< typename Value, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Array< Value, Device, Index >& v );

} // namespace Containers

} // namespace TNL

#include <TNL/Containers/Array.hpp>
