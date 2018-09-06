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
namespace DistributedContainers {

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
DistributedArray( IndexMap indexMap, CommunicationGroup group )
{
   setDistribution( indexMap, group );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
setDistribution( IndexMap indexMap, CommunicationGroup group )
{
   this->indexMap = indexMap;
   this->group = group;
   if( group != Communicator::NullGroup )
      localData.setSize( indexMap.getLocalSize() );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
const IndexMap&
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getIndexMap() const
{
   return indexMap;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
typename Communicator::CommunicationGroup
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getCommunicationGroup() const
{
   return group;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
typename DistributedArray< Value, Device, Communicator, Index, IndexMap >::LocalArrayViewType
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getLocalArrayView()
{
   return localData;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
typename DistributedArray< Value, Device, Communicator, Index, IndexMap >::ConstLocalArrayViewType
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getLocalArrayView() const
{
   return localData;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
copyFromGlobal( ConstLocalArrayViewType globalArray )
{
   TNL_ASSERT_EQ( indexMap.getGlobalSize(), globalArray.getSize(),
                  "given global array has different size than the distributed array" );

   LocalArrayViewType localView( localData );
   const IndexMap indexMap = getIndexMap();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      if( indexMap.isLocal( i ) )
         localView[ indexMap.getLocalIndex( i ) ] = globalArray[ i ];
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, indexMap.getGlobalSize(), kernel );
}


/*
 * Usual Array methods follow below.
 */

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
String
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getType()
{
   return String( "DistributedContainers::DistributedArray< " ) +
          TNL::getType< Value >() + ", " +
          Device::getDeviceType() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator>, " +
          TNL::getType< Index >() + ", " +
          IndexMap::getType() + " >";
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
String
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getTypeVirtual() const
{
   return getType();
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Array >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
setLike( const Array& array )
{
   indexMap = array.getIndexMap();
   group = array.getCommunicationGroup();
   localData.setLike( array.getLocalArrayView() );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
reset()
{
   indexMap.reset();
   group = Communicator::NullGroup;
   localData.reset();
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Index
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getSize() const
{
   return indexMap.getGlobalSize();
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
setValue( ValueType value )
{
   localData.setValue( value );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
setElement( IndexType i, ValueType value )
{
   const IndexType li = indexMap.getLocalIndex( i );
   localData.setElement( li, value );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Value
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
getElement( IndexType i ) const
{
   const IndexType li = indexMap.getLocalIndex( i );
   return localData.getElement( li );
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
__cuda_callable__
Value&
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator[]( IndexType i )
{
   const IndexType li = indexMap.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
__cuda_callable__
const Value&
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator[]( IndexType i ) const
{
   const IndexType li = indexMap.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
DistributedArray< Value, Device, Communicator, Index, IndexMap >&
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator=( const DistributedArray& array )
{
   setLike( array );
   localData = array.getLocalArrayView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Array >
DistributedArray< Value, Device, Communicator, Index, IndexMap >&
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator=( const Array& array )
{
   setLike( array );
   localData = array.getLocalArrayView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Array >
bool
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator==( const Array& array ) const
{
   // we can't run allreduce if the communication groups are different
   if( group != array.getCommunicationGroup() )
      return false;
   const bool localResult =
         indexMap == array.getIndexMap() &&
         localData == array.getLocalArrayView();
   bool result = true;
   if( group != CommunicatorType::NullGroup )
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, group );
   return result;
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Array >
bool
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator!=( const Array& array ) const
{
   return ! (*this == array);
}

template< typename Value,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
bool
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
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
          typename Communicator,
          typename Index,
          typename IndexMap >
bool
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
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
          typename Communicator,
          typename Index,
          typename IndexMap >
DistributedArray< Value, Device, Communicator, Index, IndexMap >::
operator bool() const
{
   return getSize() != 0;
}

} // namespace DistributedContainers
} // namespace TNL
