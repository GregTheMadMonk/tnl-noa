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

#include <memory>

#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Subrange.h>
#include <TNL/Containers/ByteArraySynchronizer.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
class DistributedArrayView
{
public:
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = Index;
   using LocalRangeType = Subrange< Index >;
   using LocalViewType = Containers::ArrayView< Value, Device, Index >;
   using ConstLocalViewType = Containers::ArrayView< std::add_const_t< Value >, Device, Index >;
   using ViewType = DistributedArrayView< Value, Device, Index >;
   using ConstViewType = DistributedArrayView< std::add_const_t< Value >, Device, Index >;
   using SynchronizerType = ByteArraySynchronizer< DeviceType, IndexType >;

   /**
    * \brief A template which allows to quickly obtain a \ref DistributedArrayView type with changed template parameters.
    */
   template< typename _Value,
             typename _Device = Device,
             typename _Index = Index >
   using Self = DistributedArrayView< _Value, _Device, _Index >;


   ~DistributedArrayView();

   // Initialization by raw data
   DistributedArrayView( const LocalRangeType& localRange, IndexType ghosts, IndexType globalSize, MPI_Comm group, LocalViewType localData )
   : localRange(localRange), ghosts(ghosts), globalSize(globalSize), group(group), localData(localData)
   {
      TNL_ASSERT_EQ( localData.getSize(), localRange.getSize() + ghosts,
                     "The local array size does not match the local range of the distributed array." );
      TNL_ASSERT_GE( ghosts, 0, "The ghosts count must be non-negative." );
   }

   DistributedArrayView() = default;

   // Copy-constructor does shallow copy.
   DistributedArrayView( const DistributedArrayView& ) = default;

   // "Templated copy-constructor" accepting any cv-qualification of Value
   template< typename Value_ >
   DistributedArrayView( const DistributedArrayView< Value_, Device, Index >& );

   // default move-constructor
   DistributedArrayView( DistributedArrayView&& ) = default;

   // method for rebinding (reinitialization) to raw data
   void bind( const LocalRangeType& localRange, IndexType ghosts, IndexType globalSize, MPI_Comm group, LocalViewType localData );

   // Note that you can also bind directly to DistributedArray and other types implicitly
   // convertible to DistributedArrayView.
   void bind( DistributedArrayView view );

   // binding to local array via raw pointer
   // (local range, ghosts, global size and communication group are preserved)
   template< typename Value_ >
   void bind( Value_* data, IndexType localSize );

   const LocalRangeType& getLocalRange() const;

   IndexType getGhosts() const;

   MPI_Comm getCommunicationGroup() const;

   LocalViewType getLocalView();

   ConstLocalViewType getConstLocalView() const;

   LocalViewType getLocalViewWithGhosts();

   ConstLocalViewType getConstLocalViewWithGhosts() const;

   void copyFromGlobal( ConstLocalViewType globalArray );

   // synchronizer stuff
   void setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer, int valuesPerElement = 1 );

   std::shared_ptr< SynchronizerType > getSynchronizer() const;

   int getValuesPerElement() const;

   // Note that this method is not thread-safe - only the thread which created
   // and "owns" the instance of this object can call this method.
   void startSynchronization();

   void waitForSynchronization() const;


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

      /**
       * \brief Process the lambda function \e f for each array element in interval [ \e begin, \e end).
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param begin The beginning of the array elements interval.
       * \param end The end of the array elements interval.
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f );

      /**
       * \brief Process the lambda function \e f for each array element in interval [ \e begin, \e end) for constant instances of the array.
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param begin The beginning of the array elements interval.
       * \param end The end of the array elements interval.
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f ) const;

   std::ostream& print( std::ostream& str ) const;
protected:
   LocalRangeType localRange;
   IndexType ghosts = 0;
   IndexType globalSize = 0;
   MPI_Comm group = MPI::NullGroup();
   LocalViewType localData;

   std::shared_ptr< SynchronizerType > synchronizer = nullptr;
   int valuesPerElement = 1;
};


template< typename Value,
          typename Device = Devices::Host,
          typename Index = int >
std::ostream& operator<<( std::ostream& str, const DistributedArrayView< Value, Device, Index >& view )
{
   return view.print( str );
}


} // namespace Containers
} // namespace TNL

#include "DistributedArrayView.hpp"
