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

#include <TNL/Containers/Array.h>
#include <TNL/Containers/DistributedArrayView.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedArray
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using LocalArrayType = Containers::Array< Value, Device, Index >;

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
    * \brief A template which allows to quickly obtain a \ref DistributedArray type with changed template parameters.
    */
   template< typename _Value,
             typename _Device = Device,
             typename _Index = Index,
             typename _Communicator = Communicator >
   using Self = DistributedArray< _Value, _Device, _Index, _Communicator >;


   DistributedArray() = default;

   DistributedArray( const DistributedArray& ) = default;

   DistributedArray( LocalRangeType localRange, Index globalSize, CommunicationGroup group = Communicator::AllGroup );

   void setDistribution( LocalRangeType localRange, Index globalSize, CommunicationGroup group = Communicator::AllGroup );

   const LocalRangeType& getLocalRange() const;

   CommunicationGroup getCommunicationGroup() const;

   /**
    * \brief Returns a modifiable view of the local part of the array.
    *
    * If \e begin or \e end is set to a non-zero value, a view for the
    * sub-interval `[begin, end)` is returned. Otherwise a view for whole
    * local part of the array view is returned.
    *
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   LocalViewType getLocalView();

   /**
    * \brief Returns a non-modifiable view of the local part of the array.
    *
    * If \e begin or \e end is set to a non-zero value, a view for the
    * sub-interval `[begin, end)` is returned. Otherwise a view for whole
    * local part of the array view is returned.
    *
    * \param begin The beginning of the array view sub-interval. It is 0 by
    *              default.
    * \param end The end of the array view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   ConstLocalViewType getConstLocalView() const;

   void copyFromGlobal( ConstLocalViewType globalArray );


   // Usual Array methods follow below.

   /**
    * \brief Returns a modifiable view of the array.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the array.
    */
   ConstViewType getConstView() const;

   /**
    * \brief Conversion operator to a modifiable view of the array.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the array.
    */
   operator ConstViewType() const;

   template< typename Array >
   void setLike( const Array& array );

   // Resets the array to the empty state.
   void reset();

   // Returns true if the current array size is zero.
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

   // Copy-assignment operator
   DistributedArray& operator=( const DistributedArray& array );

   template< typename Array,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Array>::value > >
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

   // TODO: serialization (save, load)

protected:
   ViewType view;
   LocalArrayType localData;
};

} // namespace Containers
} // namespace TNL

#include "DistributedArray.hpp"
