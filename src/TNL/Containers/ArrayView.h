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

   __cuda_callable__
   ArrayView() = default;

   // explicit initialization by raw data pointer and size
   __cuda_callable__
   ArrayView( Value* data, Index size );

   // Copy-constructor does shallow copy, so views can be passed-by-value into
   // CUDA kernels and they can be captured-by-value in __cuda_callable__
   // lambda functions.
   __cuda_callable__
   ArrayView( const ArrayView& ) = default;

   // "Templated copy-constructor" accepting any cv-qualification of Value
   template< typename Value_ >
   __cuda_callable__
   ArrayView( const ArrayView< Value_, Device, Index >& array )
   : data(array.getData()), size(array.getSize()) {}

   // default move-constructor
   __cuda_callable__
   ArrayView( ArrayView&& ) = default;

   // initialization from other array containers (using shallow copy)
   template< typename Value_ >  // template catches both const and non-const qualified Value
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
   Index getSize() const;

   void setElement( Index i, Value value );

   Value getElement( Index i ) const;

   __cuda_callable__
   Value& operator[]( Index i );

   __cuda_callable__
   const Value& operator[]( Index i ) const;

   template< typename Value_, typename Device_, typename Index_ >
   bool operator==( const ArrayView< Value_, Device_, Index_ >& view ) const;

   template< typename Value_, typename Device_, typename Index_ >
   bool operator!=( const ArrayView< Value_, Device_, Index_ >& view ) const;

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
