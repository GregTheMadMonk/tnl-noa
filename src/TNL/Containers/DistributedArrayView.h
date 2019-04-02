/***************************************************************************
                          DistributedArrayView.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Containers/DistributedArray.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedArrayView
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using ValueType = Value;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using LocalRangeType = Subrange< Index >;
   using LocalArrayViewType = Containers::ArrayView< Value, Device, Index >;
   using ConstLocalArrayViewType = Containers::ArrayView< typename std::add_const< Value >::type, Device, Index >;
   using HostType = DistributedArrayView< Value, Devices::Host, Index, Communicator >;
   using CudaType = DistributedArrayView< Value, Devices::Cuda, Index, Communicator >;

   __cuda_callable__
   DistributedArrayView() = default;

   // Copy-constructor does shallow copy, so views can be passed-by-value into
   // CUDA kernels and they can be captured-by-value in __cuda_callable__
   // lambda functions.
   __cuda_callable__
   DistributedArrayView( const DistributedArrayView& ) = default;

   // "Templated copy-constructor" accepting any cv-qualification of Value
   template< typename Value_ >
   __cuda_callable__
   DistributedArrayView( const DistributedArrayView< Value_, Device, Index, Communicator >& );

   // default move-constructor
   __cuda_callable__
   DistributedArrayView( DistributedArrayView&& ) = default;

   // initialization from distributed array
   template< typename Value_ >
   DistributedArrayView( DistributedArray< Value_, Device, Index, Communicator >& array );

   // this constructor will be used only when Value is const-qualified
   // (const views are initializable by const references)
   template< typename Value_ >
   DistributedArrayView( const DistributedArray< Value_, Device, Index, Communicator >& array );

   // method for rebinding (reinitialization)
   // Note that you can also bind directly to Array and other types implicitly
   // convertible to ArrayView.
   __cuda_callable__
   void bind( DistributedArrayView view );

   // binding to local array via raw pointer
   // (local range, global size and communication group are preserved)
   template< typename Value_ >
   void bind( Value_* data, IndexType localSize );


   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   DistributedArrayView& operator=( const DistributedArrayView& view );

   template< typename Array >
   DistributedArrayView& operator=( const Array& array );


   const LocalRangeType& getLocalRange() const;

   CommunicationGroup getCommunicationGroup() const;

   LocalArrayViewType getLocalArrayView();

   ConstLocalArrayViewType getLocalArrayView() const;

   void copyFromGlobal( ConstLocalArrayViewType globalArray );


   static String getType();


   /*
    * Usual ArrayView methods follow below.
    */
   void reset();

   // TODO: swap

   // Returns the *global* size
   IndexType getSize() const;

   // Sets all elements of the array to the given value
   void setValue( ValueType value );

   // Safe device-independent element setter
   void setElement( IndexType i, ValueType value );

   // Safe device-independent element getter
   ValueType getElement( IndexType i ) const;

   // Unsafe element accessor usable only from the Device
   __cuda_callable__
   ValueType& operator[]( IndexType i );

   // Unsafe element accessor usable only from the Device
   __cuda_callable__
   const ValueType& operator[]( IndexType i ) const;

   // Comparison operators
   template< typename Array >
   bool operator==( const Array& array ) const;

   template< typename Array >
   bool operator!=( const Array& array ) const;

   // Checks if there is an element with given value in this array
   bool containsValue( ValueType value ) const;

   // Checks if all elements in this array have the same given value
   bool containsOnlyValue( ValueType value ) const;

   // Returns true iff non-zero size is set
   operator bool() const;

protected:
   LocalRangeType localRange;
   IndexType globalSize = 0;
   CommunicationGroup group = Communicator::NullGroup;
   LocalArrayViewType localData;
};

} // namespace Containers
} // namespace TNL

#include "DistributedArrayView_impl.h"
