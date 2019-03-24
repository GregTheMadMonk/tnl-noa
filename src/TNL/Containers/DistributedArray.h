/***************************************************************************
                          DistributedArray.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>  // std::add_const

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Containers/Subrange.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedArray
: public Object
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using ValueType = Value;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using LocalRangeType = Subrange< Index >;
   using LocalArrayType = Containers::Array< Value, Device, Index >;
   using LocalArrayViewType = Containers::ArrayView< Value, Device, Index >;
   using ConstLocalArrayViewType = Containers::ArrayView< typename std::add_const< Value >::type, Device, Index >;
   using HostType = DistributedArray< Value, Devices::Host, Index, Communicator >;
   using CudaType = DistributedArray< Value, Devices::Cuda, Index, Communicator >;

   DistributedArray() = default;

   DistributedArray( DistributedArray& ) = default;

   DistributedArray( LocalRangeType localRange, Index globalSize, CommunicationGroup group = Communicator::AllGroup );

   void setDistribution( LocalRangeType localRange, Index globalSize, CommunicationGroup group = Communicator::AllGroup );

   const LocalRangeType& getLocalRange() const;

   CommunicationGroup getCommunicationGroup() const;

   // we return only the view so that the user cannot resize it
   LocalArrayViewType getLocalArrayView();

   ConstLocalArrayViewType getLocalArrayView() const;

   void copyFromGlobal( ConstLocalArrayViewType globalArray );


   static String getType();

   virtual String getTypeVirtual() const;

   // TODO: no getSerializationType method until there is support for serialization


   /*
    * Usual Array methods follow below.
    */
   template< typename Array >
   void setLike( const Array& array );

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

   // Copy-assignment operator
   DistributedArray& operator=( const DistributedArray& array );

   template< typename Array >
   DistributedArray& operator=( const Array& array );

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

   // TODO: serialization (save, load, boundLoad)

protected:
   LocalRangeType localRange;
   IndexType globalSize = 0;
   CommunicationGroup group = Communicator::NullGroup;
   LocalArrayType localData;

private:
   // TODO: disabled until they are implemented
   using Object::save;
   using Object::load;
   using Object::boundLoad;
};

} // namespace Containers

template< typename Value,
          typename Device,
          typename Index >
struct isArray< Containers::DistributedArray< Value, Device, Index > >
{
   static constexpr bool value = true;
};

} // namespace TNL

#include "DistributedArray_impl.h"
