/***************************************************************************
                          ArrayView.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz et al.
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Object.h>
#include <TNL/File.h>
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
: public Object
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
   __cuda_callable__
   ArrayView( Array< Element, Device, Index >& array );

   template< int Size >
   __cuda_callable__
   ArrayView( StaticArray< Size, Element >& array );


   // methods for rebinding (reinitialization)
   __cuda_callable__
   void bind( Element* data, const Index size );

   __cuda_callable__
   void bind( ArrayView& view );

   __cuda_callable__
   void bind( Array< Element, Device, Index >& array );

   template< int Size >
   __cuda_callable__
   void bind( StaticArray< Size, Element >& array );


   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   ArrayView& operator=( const ArrayView& view );


   static String getType();

   virtual String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;


   __cuda_callable__
   void swap( ArrayView& view );

   __cuda_callable__
   void reset();

   __cuda_callable__
   Index getSize() const;

   void setElement( Index i, Element value );

   Element getElement( Index i ) const;

   __cuda_callable__
   Element& operator[]( Index i );

   __cuda_callable__
   const Element& operator[]( Index i ) const;

   template< typename Device_ >
   bool operator==( const ArrayView< Element, Device_, Index >& view ) const;

   template< typename Device_ >
   bool operator!=( const ArrayView< Element, Device_, Index >& view ) const;

   void setValue( Element value );

   __cuda_callable__
   const Element* getData() const;

   __cuda_callable__
   Element* getData();

   //! Returns true if non-zero size is set.
   operator bool() const;

   //! Method for saving the object to a file as a binary data.
   bool save( File& file ) const;

   bool save( const String& fileName ) const;

   // TODO: Does load make sense for views? Shouldn't views always use boundLoad?
   bool load( File& file );

   bool load( const String& fileName );

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
