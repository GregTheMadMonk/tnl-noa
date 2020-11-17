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

#include <TNL/Containers/ArrayView.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Containers/Subrange.h>

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
   using LocalViewType = Containers::ArrayView< Value, Device, Index >;
   using ConstLocalViewType = Containers::ArrayView< std::add_const_t< Value >, Device, Index >;
   using ViewType = DistributedArrayView< Value, Device, Index, Communicator >;
   using ConstViewType = DistributedArrayView< std::add_const_t< Value >, Device, Index, Communicator >;

   /**
    * \brief A template which allows to quickly obtain a \ref DistributedArrayView type with changed template parameters.
    */
   template< typename _Value,
             typename _Device = Device,
             typename _Index = Index,
             typename _Communicator = Communicator >
   using Self = DistributedArrayView< _Value, _Device, _Index, _Communicator >;


   // Initialization by raw data
   DistributedArrayView( const LocalRangeType& localRange, IndexType globalSize, CommunicationGroup group, LocalViewType localData )
   : localRange(localRange), globalSize(globalSize), group(group), localData(localData)
   {
      TNL_ASSERT_EQ( localData.getSize(), localRange.getSize(),
                     "The local array size does not match the local range of the distributed array." );
   }

   DistributedArrayView() = default;

   // Copy-constructor does shallow copy.
   DistributedArrayView( const DistributedArrayView& ) = default;

   // "Templated copy-constructor" accepting any cv-qualification of Value
   template< typename Value_ >
   DistributedArrayView( const DistributedArrayView< Value_, Device, Index, Communicator >& );

   // default move-constructor
   DistributedArrayView( DistributedArrayView&& ) = default;

   // method for rebinding (reinitialization) to raw data
   void bind( const LocalRangeType& localRange, IndexType globalSize, CommunicationGroup group, LocalViewType localData );

   // Note that you can also bind directly to DistributedArray and other types implicitly
   // convertible to DistributedArrayView.
   void bind( DistributedArrayView view );

   // binding to local array via raw pointer
   // (local range, global size and communication group are preserved)
   template< typename Value_ >
   void bind( Value_* data, IndexType localSize );

   const LocalRangeType& getLocalRange() const;

   CommunicationGroup getCommunicationGroup() const;

   LocalViewType getLocalView();

   ConstLocalViewType getConstLocalView() const;

   void copyFromGlobal( ConstLocalViewType globalArray );


   /*
    * Usual ArrayView methods follow below.
    */

   /**
    * \brief Returns a modifiable view of the array view.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the array view.
    */
   ConstViewType getConstView() const;

   // Resets the array view to the empty state.
   void reset();

   // Returns true if the current array view size is zero.
   bool empty() const;

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

   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   DistributedArrayView& operator=( const DistributedArrayView& view );

   template< typename Array,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Array>::value > >
   DistributedArrayView& operator=( const Array& array );

   // Comparison operators
   template< typename Array >
   bool operator==( const Array& array ) const;

   template< typename Array >
   bool operator!=( const Array& array ) const;

   // Checks if there is an element with given value in this array
   bool containsValue( ValueType value ) const;

   // Checks if all elements in this array have the same given value
   bool containsOnlyValue( ValueType value ) const;

protected:
   LocalRangeType localRange;
   IndexType globalSize = 0;
   CommunicationGroup group = Communicator::NullGroup;
   LocalViewType localData;
};

} // namespace Containers
} // namespace TNL

#include "DistributedArrayView.hpp"
