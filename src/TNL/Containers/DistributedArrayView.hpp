/***************************************************************************
                          DistributedArrayView_impl.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedArrayView.h"

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Value_ >
DistributedArrayView< Value, Device, Index, Communicator >::
DistributedArrayView( const DistributedArrayView< Value_, Device, Index, Communicator >& view )
: localRange( view.getLocalRange() ),
  globalSize( view.getSize() ),
  group( view.getCommunicationGroup() ),
  localData( view.getConstLocalView() )
{}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
bind( const LocalRangeType& localRange, IndexType globalSize, CommunicationGroup group, LocalViewType localData )
{
   TNL_ASSERT_EQ( localData.getSize(), localRange.getSize(),
                  "The local array size does not match the local range of the distributed array." );

   this->localRange = localRange;
   this->globalSize = globalSize;
   this->group = group;
   this->localData.bind( localData );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
bind( DistributedArrayView view )
{
   localRange = view.getLocalRange();
   globalSize = view.getSize();
   group = view.getCommunicationGroup();
   localData.bind( view.getLocalView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Value_ >
void
DistributedArrayView< Value, Device, Index, Communicator >::
bind( Value_* data, IndexType localSize )
{
   TNL_ASSERT_EQ( localSize, localRange.getSize(),
                  "The local array size does not match the local range of the distributed array." );
   localData.bind( data, localSize );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
const Subrange< Index >&
DistributedArrayView< Value, Device, Index, Communicator >::
getLocalRange() const
{
   return localRange;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename Communicator::CommunicationGroup
DistributedArrayView< Value, Device, Index, Communicator >::
getCommunicationGroup() const
{
   return group;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArrayView< Value, Device, Index, Communicator >::LocalViewType
DistributedArrayView< Value, Device, Index, Communicator >::
getLocalView()
{
   return localData;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArrayView< Value, Device, Index, Communicator >::ConstLocalViewType
DistributedArrayView< Value, Device, Index, Communicator >::
getConstLocalView() const
{
   return localData;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
copyFromGlobal( ConstLocalViewType globalArray )
{
   TNL_ASSERT_EQ( getSize(), globalArray.getSize(),
                  "given global array has different size than the distributed array view" );

   LocalViewType localView( localData );
   const LocalRangeType localRange = getLocalRange();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      localView[ i ] = globalArray[ localRange.getGlobalIndex( i ) ];
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localRange.getSize(), kernel );
}


template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArrayView< Value, Device, Index, Communicator >::ViewType
DistributedArrayView< Value, Device, Index, Communicator >::
getView()
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArrayView< Value, Device, Index, Communicator >::ConstViewType
DistributedArrayView< Value, Device, Index, Communicator >::
getConstView() const
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
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
bool
DistributedArrayView< Value, Device, Index, Communicator >::
empty() const
{
   return getSize() == 0;
}

// TODO: swap

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Index
DistributedArrayView< Value, Device, Index, Communicator >::
getSize() const
{
   return globalSize;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
setValue( ValueType value )
{
   localData.setValue( value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArrayView< Value, Device, Index, Communicator >::
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
DistributedArrayView< Value, Device, Index, Communicator >::
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
DistributedArrayView< Value, Device, Index, Communicator >::
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
DistributedArrayView< Value, Device, Index, Communicator >::
operator[]( IndexType i ) const
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArrayView< Value, Device, Index, Communicator >&
DistributedArrayView< Value, Device, Index, Communicator >::
operator=( const DistributedArrayView& view )
{
   TNL_ASSERT_EQ( getSize(), view.getSize(), "The sizes of the array views must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getLocalRange(), view.getLocalRange(), "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getCommunicationGroup(), view.getCommunicationGroup(), "The communication groups of the array views must be equal." );
   localData = view.getConstLocalView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array, typename..., typename >
DistributedArrayView< Value, Device, Index, Communicator >&
DistributedArrayView< Value, Device, Index, Communicator >::
operator=( const Array& array )
{
   TNL_ASSERT_EQ( getSize(), array.getSize(), "The global sizes must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getLocalRange(), array.getLocalRange(), "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getCommunicationGroup(), array.getCommunicationGroup(), "The communication groups must be equal." );
   localData = array.getConstLocalView();
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array >
bool
DistributedArrayView< Value, Device, Index, Communicator >::
operator==( const Array& array ) const
{
   // we can't run allreduce if the communication groups are different
   if( group != array.getCommunicationGroup() )
      return false;
   const bool localResult =
         localRange == array.getLocalRange() &&
         globalSize == array.getSize() &&
         localData == array.getConstLocalView();
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
DistributedArrayView< Value, Device, Index, Communicator >::
operator!=( const Array& array ) const
{
   return ! (*this == array);
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArrayView< Value, Device, Index, Communicator >::
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
DistributedArrayView< Value, Device, Index, Communicator >::
containsOnlyValue( ValueType value ) const
{
   bool result = true;
   if( group != CommunicatorType::NullGroup ) {
      const bool localResult = localData.containsOnlyValue( value );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, group );
   }
   return result;
}

} // namespace Containers
} // namespace TNL
