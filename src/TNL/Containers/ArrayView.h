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

#include <TNL/TypeTraits.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {

template< typename Value, typename Device, typename Index >
class Array;

template< int Size, typename Value >
class StaticArray;

/**
 * \brief ArrayView serves for accessing array of data allocated by TNL::Array or
 * another way. It makes no data deallocation at the end of its life cycle. Compared
 * to TNL Array, it is lighter data structure and therefore it is more efficient
 * especially when it is being passed on GPU. The ArrayView can also be created
 * in CUDA kernels which is not the case of Array.
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
 * allocated on host (CPU). If the array is allocated on GPU, the operator[]
 * can be called only from kernels running on the device (GPU). On the other
 * hand, methods \ref setElement and \ref getElement, can be called only from the
 * host (CPU) does not matter if the array resides on the host or the device.
 * In the latter case, explicit data transfer between host and device (via PCI
 * express or NVlink in more lucky systems) is invoked and so it can be very
 * slow. In not time critical parts of code, this is not an issue, however.
 * Another way to change data being accessed by the ArrayView is \ref evaluate which evaluates
 * given lambda function. This is performed at the same place where the array is
 * allocated i.e. it is efficient even on GPU. For simple checking of the array
 * contents, one may use methods \ref containValue and \ref containsValue and
 * \ref containsOnlyValue.
 *
 * \par Example
 * \include ArrayViewExample.cpp
 *
 * See also \ref Containers::Arrav, \ref Containers::Vector, \ref Containers::VectorView,
 * Containers::StaticArray, Containers::StaticVector.
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
class ArrayView
{
public:
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = ArrayView< Value, Devices::Host, Index >;
   using CudaType = ArrayView< Value, Devices::Cuda, Index >;
   using ThisType = ArrayView< Value, Device, Index >;
   using SerializationType = Array< Value, Devices::Host, Index >;

   /**
    * \brief Returns type of array view in C++ style.
    *
    * \return String with array view type.
    */
   static String getType();

   /**
    * \brief Basic constructor for empty ArrayView.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   ArrayView() = default;

   /**
    * \brief Constructor with explicit initialization by raw data pointer and size.
    *
    * This method can be called from device kernels.
    *
    * \param data is data pointer
    * \param size is number of elements to be managed by the array view
    */
   __cuda_callable__
   ArrayView( Value* data, Index size );

   /**
    * \brief Copy constructor.
    * Copy-constructor does shallow copy, so views can be passed-by-value into
    * CUDA kernels and they can be captured-by-value in __cuda_callable__
    * lambda functions.
    *
    * This method can be called from device kernels.
    *
    * \param view is ArrayView to be copied.
    */
   __cuda_callable__
   ArrayView( const ArrayView& view ) = default;

   /**
    * \brief "Templated copy-constructor".
    *
    * It makes shallow copy only.
    *
    * This method can be called from device kernels.
    *
    * \tparam Value is any cv-qualified ValueType.
    */
   template< typename Value_ >
   __cuda_callable__
   ArrayView( const ArrayView< Value_, Device, Index >& array )
   : data(array.getData()), size(array.getSize()) {}

   /**
    * \brief Default move-constructor.
    *
    * This method can be called from device kernels.
    *
    * \param view is ArrayView to be moved to this ArrayView.
    */
   __cuda_callable__
   ArrayView( ArrayView&& view ) = default;

   /**
    * \brief Constructor for initialization from other array containers.
    *
    * It makes shallow copy only.
    *
    * This method can be called from device kernels.
    *
    * \tparam Value_ can be both const and non-const qualified Value.
    */
   template< typename Value_ >
   __cuda_callable__
   ArrayView( Array< Value_, Device, Index >& array );

   /**
    * \brief Constructor for initialization with static array.
    *
    * This method can be called from device kernels.
    *
    * \tparam Size is size of the static array.
    * \tparam Value_ can be both const and non-const qualified Value.
    *
    * \param array is a static array the array view is initialized with.
    */
   template< int Size, typename Value_ >
   __cuda_callable__
   ArrayView( StaticArray< Size, Value_ >& array );

   /**
    * \brief Copy constructor from constant Array.
    *
    * This constructor will be used only when Value is const-qualified
    * (const views are initializable by const references).
    *
    * This method can be called from device kernels.
    *
    * \tparam Value_ can be both const and non-const qualified Value
    * \param array is an array the array view is initialized with.
    */
   template< typename Value_ >
   __cuda_callable__
   ArrayView( const Array< Value_, Device, Index >& array );

   /**
    * \brief Constructor for initialization with static array.
    *
    * This method can be called from device kernels.
    *
    * \tparam Size is size of the static array.
    * \tparam Value_ can be both const and non-const qualified Value.
    *
    * \param array is a static array the array view is initialized with.
    */
   template< int Size, typename Value_ >  // template catches both const and non-const qualified Value
   __cuda_callable__
   ArrayView( const StaticArray< Size, Value_ >& array );

   /**
    * \brief Method for rebinding (reinitialization).
    *
    * This method can be called from device kernels.
    *
    * \param data is pointer to data to be bound to the array view.
    * \param size is the number of elements to be managed by the array view.
    */
   __cuda_callable__
   void bind( Value* data, const Index size );

   /**
    * \brief Method for rebinding (reinitialization) with another ArrayView.
    *
    * Note that you can also bind directly to Array and other types implicitly
    * convertible to ArrayView.
    *
    * This method can be called from device kernels.
    *
    * \param view is array view to be bound.
    */
   __cuda_callable__
   void bind( ArrayView view );

   /**
    * \brief Assignment operator.
    *
    * Copy-assignment does deep copy, just like regular array, but the sizes
    * must match (i.e. copy-assignment cannot resize).
    *
    * \param view is array view to be copied
    */
   ArrayView& operator=( const ArrayView& view );

   /**
    * \brief Assignment operator for array-like containers or single value.
    *
    * If \e T is array type i.e. \ref Array, \ref ArrayView, \ref StaticArray,
    * \ref Vector, \ref VectorView, \ref StaticVector, \ref DistributedArray,
    * \ref DistributedArrayView, \ref DistributedVector or
    * \ref DistributedVectorView, its elements are copied into this array. If
    * it is other type convertibly to ArrayView::ValueType, all array elements are
    * set to the value \e data.
    *
    * \tparam T is type of array or value type.
    *
    * \param data is a reference to array or value.
    *
    * \return Reference to this array.
    */
   template< typename T >
   ArrayView& operator=( const T& array );

   /**
    * \brief Swaps this array view content with another.
    *
    * The swap is done in a shallow way, i.e. swapping only  pointers and sizes.
    *
    * This method can be called from device kernels.
    *
    * \param view is the array view to be swapped with this array view.
    */
   __cuda_callable__
   void swap( ArrayView& view );

   /***
    * \brief Resets the array view.
    *
    * The array view behaves like being empty after calling this method.
    *
    * This method can be called from device kernels.
    */
   __cuda_callable__
   void reset();

   /**
    * \brief Returns constant pointer to data managed by the array view.
    *
    * This method can be called from device kernels.
    *
    * \return Pointer to array data.
    */
   __cuda_callable__
   const Value* getData() const;

   /**
    * \brief Returns pointer to data managed by the array view.
    *
    * This method can be called from device kernels.
    *
    * \return Pointer to array data.
    */
   __cuda_callable__
   Value* getData();

   /**
    * \brief Returns constant pointer to data managed by the array view.
    *
    * Use this method in algorithms where you want to emphasize that
    * C-style array pointer is required.
    *
    * This method can be called from device kernels.
    *
    * \return Pointer to array data.
    */
   __cuda_callable__
   const Value* getArrayData() const;

   /**
    * \brief Returns pointer to data managed by the array view.
    *
    * Use this method in algorithms where you want to emphasize that
    * C-style array pointer is required.
    *
    * This method can be called from device kernels.
    *
    * \return Pointer to array data.
    */
   __cuda_callable__
   Value* getArrayData();

   /**
    * \brief Returns the array view size, i.e. number of elements being managed by the array view.
    *
    * This method can be called from device kernels.
    *
    * \return The array view size.
    */
   __cuda_callable__
   Index getSize() const;

   /**
    * \brief Array view elements setter - change value of an element at position \e i.
    *
    * This method can be called only from the host system (CPU) but even for
    * array views managing data allocated on device (GPU).
    *
    * \param i is element index.
    * \param v is the new value of the element.
    */
   void setElement( Index i, Value value );

   /**
    * \brief Array view elements getter - returns value of an element at position \e i.
    *
    * This method can be called only from the host system (CPU) but even for
    * array views managing data allocated on device (GPU).
    *
    * \param i Index position of an element.
    *
    * \return Copy of i-th element.
    */
   Value getElement( Index i ) const;

   /**
    * \brief Accesses specified element at the position \e i.
    *
    * This method can be called from device (GPU) kernels if the array view
    * manages data allocated on the device. In this case, it cannot be called
    * from the host (CPU.)
    *
    * \param i is position of the element.
    *
    * \return Reference to i-th element.
    */
   __cuda_callable__
   Value& operator[]( Index i );

   /**
    * \brief Returns constant reference to an element at a position \e i.
    *
    * This method can be called from device (GPU) kernels if the array view
    * manages data allocated on the device. In this case, it cannot be called
    * from the host (CPU.)
    *
    * \param i is position of the element.
    *
    * \return Reference to i-th element.
    */
   __cuda_callable__
   const Value& operator[]( Index i ) const;

   /**
    * \brief Comparison operator with another array view \e view.
    *
    * \tparam Value_ is the value type of the right-hand-side array view.
    * \tparam Device_ is the device type of the right-hand-side array view.
    * \tparam Index_ is the index type of the right-hand-side array view.
    * \param  view is reference to the right-hand-side array view.
    *
    * \return True if both array views are equal element-wise and false otherwise.
    */
   template< typename Value_, typename Device_, typename Index_ >
   bool operator==( const ArrayView< Value_, Device_, Index_ >& view ) const;

   /**
    * \brief Comparison operator with another array-like container \e array.
    *
    * \tparam ArrayT is type of an array-like container, i.e Array, ArrayView, Vector, VectorView, DistributedArray, DistributedVector etc.
    * \param array is reference to an array.
    *
    * \return True if both array views are equal element-wise and false otherwise.
    */
   template< typename ArrayT >
   bool operator == ( const ArrayT& array ) const;

   /**
    * \brief Comparison negation operator with another array view \e view.
    *
    * \tparam Value_ is the value type of the right-hand-side array view.
    * \tparam Device_ is the device type of the right-hand-side array view.
    * \tparam Index_ is the index type of the right-hand-side array view.
    * \param  view is reference to the right-hand-side array view.
    *
    * \return True if both array views are not equal element-wise and false otherwise.
    */
   template< typename Value_, typename Device_, typename Index_ >
   bool operator!=( const ArrayView< Value_, Device_, Index_ >& view ) const;

   /**
    * \brief Comparison negation operator with another array-like container \e array.
    *
    * \tparam ArrayT is type of an array-like container, i.e Array, ArrayView, Vector, VectorView, DistributedArray, DistributedVector etc.
    * \param array is reference to an array.
    *
    * \return True if both array views are not equal element-wise and false otherwise.
    */
   template< typename ArrayT >
   bool operator != ( const ArrayT& array ) const;

   /**
    * \brief Sets the array view elements to given value.
    *
    * Sets all the array values to \e v.
    *
    * \param v Reference to a value.
    */
   void setValue( Value value );

   /**
    * \brief Sets the array elements using given lambda function.
    *
    * Sets all the array values to \e v.
    *
    * \param v Reference to a value.
    */
   template< typename Function >
   void evaluate( Function& f,
                  const Index begin = 0,
                  Index end = -1 );

   /**
    * \brief Checks if there is an element with value \e v.
    *
    * By default, the method checks all array view elements. By setting indexes
    * \e begin and \e end, only elements in given interval are checked.
    *
    * \param v is reference to the value.
    * \param begin is the first element to be checked
    * \param end is the last element to be checked. If \e end equals -1, its
    * value is replaces by the array size.
    *
    * \return True if there is **at least one** array element in interval [\e begin, \e end ) having value \e v.
    */
   bool containsValue( Value value ) const;

   /**
    * \brief Checks if all elements have the same value \e v.
    *
    * By default, the method checks all array view elements. By setting indexes
    * \e begin and \e end, only elements in given interval are checked.
    *
    * \param v Reference to a value.
    * \param begin is the first element to be checked
    * \param end is the last element to be checked. If \e end equals -1, its
    * value is replaces by the array size.
    *
    * \return True if there **all** array elements in interval [\e begin, \e end ) have value \e v.
    */
   bool containsOnlyValue( Value value ) const;

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
    * \brief Method for saving the array view to a file as a binary data.
    *
    * \param fileName String defining the name of a file.
    */
   void save( const String& fileName ) const;

   /**
    * \brief Method for restoring the array view from a file.
    *
    * \param fileName String defining the name of a file.
    */
   void load( const String& fileName );


protected:
   //! Pointer to allocated data
   Value* data = nullptr;

   //! Number of allocated elements
   Index size = 0;
};

template< typename Value, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Value, Device, Index >& v );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/ArrayView.hpp>
