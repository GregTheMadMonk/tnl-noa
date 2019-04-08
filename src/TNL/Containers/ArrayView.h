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

   template< int Size, typename Value_ >  // template catches both const and non-const qualified Value
   __cuda_callable__
   ArrayView( StaticArray< Size, Value_ >& array );

   // these constructors will be used only when Value is const-qualified
   // (const views are initializable by const references)
   template< typename Value_ >  // template catches both const and non-const qualified Value
   __cuda_callable__
   ArrayView( const Array< Value_, Device, Index >& array );

   template< int Size, typename Value_ >  // template catches both const and non-const qualified Value
   __cuda_callable__
   ArrayView( const StaticArray< Size, Value_ >& array );


   // methods for rebinding (reinitialization)
   __cuda_callable__
   void bind( Value* data, const Index size );

   // Note that you can also bind directly to Array and other types implicitly
   // convertible to ArrayView.
   __cuda_callable__
   void bind( ArrayView view );


   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   ArrayView& operator=( const ArrayView& view );

   template< typename Array >
   ArrayView& operator=( const Array& array );

   static String getType();

   __cuda_callable__
   void swap( ArrayView& view );

   __cuda_callable__
   void reset();

   __cuda_callable__
   const Value* getData() const;

   __cuda_callable__
   Value* getData();

   __cuda_callable__
   const Value* getArrayData() const;

   __cuda_callable__
   Value* getArrayData();

   __cuda_callable__
   Index getSize() const;

   void setElement( Index i, Value value );

   Value getElement( Index i ) const;

   __cuda_callable__
   Value& operator[]( Index i );

   __cuda_callable__
   const Value& operator[]( Index i ) const;

   template< typename Value_, typename Device_, typename Index_ >
   bool operator==( const ArrayView< Value_, Device_, Index_ >& view ) const;

   template< typename ArrayT >
   bool operator == ( const ArrayT& array ) const;   

   template< typename Value_, typename Device_, typename Index_ >
   bool operator!=( const ArrayView< Value_, Device_, Index_ >& view ) const;
   
   template< typename ArrayT >
   bool operator != ( const ArrayT& array ) const;

   void setValue( Value value );

   // Checks if there is an element with given value in this array
   bool containsValue( Value value ) const;

   // Checks if all elements in this array have the same given value
   bool containsOnlyValue( Value value ) const;

   //! Returns true if non-zero size is set.
   operator bool() const;

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

#include <TNL/Containers/ArrayView_impl.h>
