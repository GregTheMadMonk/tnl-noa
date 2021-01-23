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
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class DistributedArray
{
   using LocalArrayType = Containers::Array< Value, Device, Index, Allocator >;

public:
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = Index;
   using AllocatorType = Allocator;
   using LocalRangeType = Subrange< Index >;
   using LocalViewType = Containers::ArrayView< Value, Device, Index >;
   using ConstLocalViewType = Containers::ArrayView< std::add_const_t< Value >, Device, Index >;
   using ViewType = DistributedArrayView< Value, Device, Index >;
   using ConstViewType = DistributedArrayView< std::add_const_t< Value >, Device, Index >;
   using SynchronizerType = typename ViewType::SynchronizerType;

   /**
    * \brief A template which allows to quickly obtain a \ref DistributedArray type with changed template parameters.
    */
   template< typename _Value,
             typename _Device = Device,
             typename _Index = Index,
             typename _Allocator = typename Allocators::Default< _Device >::template Allocator< _Value > >
   using Self = DistributedArray< _Value, _Device, _Index, _Allocator >;


   ~DistributedArray();

   /**
    * \brief Constructs an empty array with zero size.
    */
   DistributedArray() = default;

   /**
    * \brief Constructs an empty array and sets the provided allocator.
    *
    * \param allocator The allocator to be associated with this array.
    */
   explicit DistributedArray( const AllocatorType& allocator );

   /**
    * \brief Copy constructor (makes a deep copy).
    *
    * \param array The array to be copied.
    */
   explicit DistributedArray( const DistributedArray& array );

   /**
    * \brief Copy constructor with a specific allocator (makes a deep copy).
    *
    * \param array The array to be copied.
    * \param allocator The allocator to be associated with this array.
    */
   explicit DistributedArray( const DistributedArray& array, const AllocatorType& allocator );

   DistributedArray( LocalRangeType localRange, Index ghosts, Index globalSize, MPI_Comm group = MPI::AllGroup(), const AllocatorType& allocator = AllocatorType() );

   void setDistribution( LocalRangeType localRange, Index ghosts, Index globalSize, MPI_Comm group = MPI::AllGroup() );

   const LocalRangeType& getLocalRange() const;

   IndexType getGhosts() const;

   MPI_Comm getCommunicationGroup() const;

   AllocatorType getAllocator() const;

   /**
    * \brief Returns a modifiable view of the local part of the array.
    */
   LocalViewType getLocalView();

   /**
    * \brief Returns a non-modifiable view of the local part of the array.
    */
   ConstLocalViewType getConstLocalView() const;

   /**
    * \brief Returns a modifiable view of the local part of the array,
    * including ghost values.
    */
   LocalViewType getLocalViewWithGhosts();

   /**
    * \brief Returns a non-modifiable view of the local part of the array,
    * including ghost values.
    */
   ConstLocalViewType getConstLocalViewWithGhosts() const;

   void copyFromGlobal( ConstLocalViewType globalArray );

   // synchronizer stuff
   void setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer, int valuesPerElement = 1 );

   std::shared_ptr< SynchronizerType > getSynchronizer() const;

   int getValuesPerElement() const;

   void startSynchronization();

   void waitForSynchronization() const;


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

private:
   template< typename Array, std::enable_if_t< std::is_same< typename Array::DeviceType, DeviceType >::value, bool > = true >
   static void setSynchronizerHelper( ViewType& view, const Array& array )
   {
      view.setSynchronizer( array.getSynchronizer(), array.getValuesPerElement() );
   }
   template< typename Array, std::enable_if_t< ! std::is_same< typename Array::DeviceType, DeviceType >::value, bool > = true >
   static void setSynchronizerHelper( ViewType& view, const Array& array ) {}
};

} // namespace Containers
} // namespace TNL

#include "DistributedArray.hpp"
