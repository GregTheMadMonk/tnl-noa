/***************************************************************************
                          DistributedArray_impl.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedArray.h"

#include <TNL/ParallelFor.h>
#include <TNL/Communicators/MpiDefs.h>  // important only when MPI is disabled

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
DistributedArray( LocalRangeType localRange, IndexType globalSize, CommunicationGroup group )
{
   setDistribution( localRange, globalSize, group );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setDistribution( LocalRangeType localRange, IndexType globalSize, CommunicationGroup group )
{
   TNL_ASSERT_LE( localRange.getEnd(), globalSize, "end of the local range is outside of the global range" );
   this->localRange = localRange;
   this->globalSize = globalSize;
   this->group = group;
   if( group != Communicator::NullGroup )
      localData.setSize( localRange.getSize() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
const Subrange< Index >&
DistributedArray< Value, Device, Index, Communicator >::
getLocalRange() const
{
   return localRange;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename Communicator::CommunicationGroup
DistributedArray< Value, Device, Index, Communicator >::
getCommunicationGroup() const
{
   return group;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::LocalArrayViewType
DistributedArray< Value, Device, Index, Communicator >::
getLocalArrayView()
{
   return localData.getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ConstLocalArrayViewType
DistributedArray< Value, Device, Index, Communicator >::
getLocalArrayView() const
{
   return localData.getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
copyFromGlobal( ConstLocalArrayViewType globalArray )
{
   TNL_ASSERT_EQ( getSize(), globalArray.getSize(),
                  "given global array has different size than the distributed array" );

   LocalArrayViewType localView( localData );
   const LocalRangeType localRange = getLocalRange();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      localView[ i ] = globalArray[ localRange.getGlobalIndex( i ) ];
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, localRange.getSize(), kernel );
}


/*
 * Usual Array methods follow below.
 */

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ViewType
DistributedArray< Value, Device, Index, Communicator >::
getView()
{
   return ViewType( getLocalRange(), getSize(), getCommunicationGroup(), getLocalArrayView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ConstViewType
DistributedArray< Value, Device, Index, Communicator >::
getConstView() const
{
   return ConstViewType( getLocalRange(), getSize(), getCommunicationGroup(), getLocalArrayView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
String
DistributedArray< Value, Device, Index, Communicator >::
getType()
{
   return String( "Containers::DistributedArray< " ) +
          TNL::getType< Value >() + ", " +
          Device::getDeviceType() + ", " +
          TNL::getType< Index >() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator> >";
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
String
DistributedArray< Value, Device, Index, Communicator >::
getTypeVirtual() const
{
   return getType();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array >
void
DistributedArray< Value, Device, Index, Communicator >::
setLike( const Array& array )
{
   localRange = array.getLocalRange();
   globalSize = array.getSize();
   group = array.getCommunicationGroup();
   localData.setLike( array.getLocalArrayView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
reset()
{
   localRange.reset();
   globalSize = 0;
   group = Communicator::NullGroup;
   localData.reset();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Index
DistributedArray< Value, Device, Index, Communicator >::
getSize() const
{
   return globalSize;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setValue( ValueType value )
{
   localData.setValue( value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setElement( IndexType i, ValueType value )
{
   const IndexType li = localRange.getLocalIndex( i );
   localData.setElement( li, value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Value
DistributedArray< Value, Device, Index, Communicator >::
getElement( IndexType i ) const
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData.getElement( li );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
__cuda_callable__
Value&
DistributedArray< Value, Device, Index, Communicator >::
operator[]( IndexType i )
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
__cuda_callable__
const Value&
DistributedArray< Value, Device, Index, Communicator >::
operator[]( IndexType i ) const
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >&
DistributedArray< Value, Device, Index, Communicator >::
operator=( const DistributedArray& array )
{
   setLike( array );
   localData = array.getLocalArrayView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array >
DistributedArray< Value, Device, Index, Communicator >&
DistributedArray< Value, Device, Index, Communicator >::
operator=( const Array& array )
{
   setLike( array );
   localData = array.getLocalArrayView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array >
bool
DistributedArray< Value, Device, Index, Communicator >::
operator==( const Array& array ) const
{
   // we can't run allreduce if the communication groups are different
   if( group != array.getCommunicationGroup() )
      return false;
   const bool localResult =
         localRange == array.getLocalRange() &&
         globalSize == array.getSize() &&
         localData == array.getLocalArrayView();
   bool result = true;
   if( group != CommunicatorType::NullGroup )
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, group );
   return result;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array >
bool
DistributedArray< Value, Device, Index, Communicator >::
operator!=( const Array& array ) const
{
   return ! (*this == array);
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArray< Value, Device, Index, Communicator >::
containsValue( ValueType value ) const
{
   bool result = false;
   if( group != CommunicatorType::NullGroup ) {
      const bool localResult = localData.containsValue( value );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, group );
   }
   return result;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArray< Value, Device, Index, Communicator >::
containsOnlyValue( ValueType value ) const
{
   bool result = true;
   if( group != CommunicatorType::NullGroup ) {
      const bool localResult = localData.containsOnlyValue( value );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, group );
   }
   return result;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
operator bool() const
{
   return getSize() != 0;
}

} // namespace Containers
} // namespace TNL
