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

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {

template< typename Element, typename Device, typename Index >
class Array;

template< int Size, typename Element >
class StaticArray;

template< typename Element,
          typename Device = Devices::Host,
          typename Index = int >
class ArrayView
{
public:
   using ElementType = Element;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = ArrayView< Element, Devices::Host, Index >;
   using CudaType = ArrayView< Element, Devices::Cuda, Index >;

   __cuda_callable__
   ArrayView() = default;

   // explicit initialization by raw data pointer and size
   __cuda_callable__
   ArrayView( Element* data, Index size );

   // Copy-constructor does shallow copy, so views can be passed-by-value into
   // CUDA kernels and they can be captured-by-value in __cuda_callable__
   // lambda functions.
   __cuda_callable__
   ArrayView( const ArrayView& ) = default;

   // default move-constructor
   __cuda_callable__
   ArrayView( ArrayView&& ) = default;

   // initialization from other array containers (using shallow copy)
   template< typename Element_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   ArrayView( Array< Element_, Device, Index >& array );

   template< int Size, typename Element_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   ArrayView( StaticArray< Size, Element_ >& array );

   // these constructors will be used only when Element is const-qualified
   // (const views are initializable by const references)
   template< typename Element_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   ArrayView( const Array< Element_, Device, Index >& array );

   template< int Size, typename Element_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   ArrayView( const StaticArray< Size, Element_ >& array );


   // methods for rebinding (reinitialization)
   __cuda_callable__
   void bind( Element* data, const Index size );

   // Note that you can also bind directly to Array and other types implicitly
   // convertible to ArrayView.
   __cuda_callable__
   void bind( ArrayView view );


   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   ArrayView& operator=( const ArrayView& view );

   template< typename Element_, typename Device_, typename Index_ >
   ArrayView& operator=( const ArrayView< Element_, Device_, Index_ >& view );


   static String getType();


   __cuda_callable__
   void swap( ArrayView& view );

   __cuda_callable__
   void reset();

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   Element* getData();

   __cuda_callable__
   Index getSize() const;

   void setElement( Index i, Element value );

   Element getElement( Index i ) const;

   __cuda_callable__
   Element& operator[]( Index i );

   __cuda_callable__
   const Element& operator[]( Index i ) const;

   template< typename Element_, typename Device_, typename Index_ >
   bool operator==( const ArrayView< Element_, Device_, Index_ >& view ) const;

   template< typename Element_, typename Device_, typename Index_ >
   bool operator!=( const ArrayView< Element_, Device_, Index_ >& view ) const;

   void setValue( Element value );

   // Checks if there is an element with given value in this array
   bool containsValue( Element value ) const;

   // Checks if all elements in this array have the same given value
   bool containsOnlyValue( Element value ) const;

   //! Returns true if non-zero size is set.
   operator bool() const;

protected:
   //! Pointer to allocated data
   Element* data = nullptr;

   //! Number of allocated elements
   Index size = 0;
};

template< typename Element, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const ArrayView< Element, Device, Index >& v );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/ArrayView_impl.h>
