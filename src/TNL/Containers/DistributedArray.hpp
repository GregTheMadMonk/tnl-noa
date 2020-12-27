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

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Communicators/MpiDefs.h>  // important only when MPI is disabled

namespace TNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
~DistributedArray()
{
   // Wait for pending async operation, otherwise the synchronizer would crash
   // if the array goes out of scope.
   waitForSynchronization();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
DistributedArray( const DistributedArray& array )
{
   setLike( array );
   localData = array.getConstLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedArray< Value, Device, Index, Communicator >::
DistributedArray( LocalRangeType localRange, IndexType ghosts, IndexType globalSize, CommunicationGroup group )
{
   setDistribution( localRange, ghosts, globalSize, group );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setDistribution( LocalRangeType localRange, IndexType ghosts, IndexType globalSize, CommunicationGroup group )
{
   TNL_ASSERT_LE( localRange.getEnd(), globalSize, "end of the local range is outside of the global range" );
   if( group != Communicator::NullGroup )
      localData.setSize( localRange.getSize() + ghosts );
   view.bind( localRange, ghosts, globalSize, group, localData.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
const Subrange< Index >&
DistributedArray< Value, Device, Index, Communicator >::
getLocalRange() const
{
   return view.getLocalRange();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Index
DistributedArray< Value, Device, Index, Communicator >::
getGhosts() const
{
   return view.getGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename Communicator::CommunicationGroup
DistributedArray< Value, Device, Index, Communicator >::
getCommunicationGroup() const
{
   return view.getCommunicationGroup();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::LocalViewType
DistributedArray< Value, Device, Index, Communicator >::
getLocalView()
{
   return view.getLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ConstLocalViewType
DistributedArray< Value, Device, Index, Communicator >::
getConstLocalView() const
{
   return view.getConstLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::LocalViewType
DistributedArray< Value, Device, Index, Communicator >::
getLocalViewWithGhosts()
{
   return view.getLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ConstLocalViewType
DistributedArray< Value, Device, Index, Communicator >::
getConstLocalViewWithGhosts() const
{
   return view.getConstLocalViewWithGhosts();
}


template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
copyFromGlobal( ConstLocalViewType globalArray )
{
   view.copyFromGlobal( globalArray );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer, int valuesPerElement )
{
   view.setSynchronizer( synchronizer, valuesPerElement );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
std::shared_ptr< typename DistributedArrayView< Value, Device, Index, Communicator >::SynchronizerType >
DistributedArray< Value, Device, Index, Communicator >::
getSynchronizer() const
{
   return view.getSynchronizer();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
int
DistributedArray< Value, Device, Index, Communicator >::
getValuesPerElement() const
{
   return view.getValuesPerElement();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
startSynchronization()
{
   view.startSynchronization();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
waitForSynchronization() const
{
   view.waitForSynchronization();
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
   return view;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedArray< Value, Device, Index, Communicator >::ConstViewType
DistributedArray< Value, Device, Index, Communicator >::
getConstView() const
{
   return view.getConstView();
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
   template< typename Array >
void
DistributedArray< Value, Device, Index, Communicator >::
setLike( const Array& array )
{
   localData.setLike( array.getConstLocalViewWithGhosts() );
   view.bind( array.getLocalRange(), array.getGhosts(), array.getSize(), array.getCommunicationGroup(), localData.getView() );
   // set, but do not unset, the synchronizer
   if( array.getSynchronizer() )
      setSynchronizerHelper( view, array );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
reset()
{
   view.reset();
   localData.reset();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArray< Value, Device, Index, Communicator >::
empty() const
{
   return view.empty();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Index
DistributedArray< Value, Device, Index, Communicator >::
getSize() const
{
   return view.getSize();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setValue( ValueType value )
{
   view.setValue( value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedArray< Value, Device, Index, Communicator >::
setElement( IndexType i, ValueType value )
{
   view.setElement( i, value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
Value
DistributedArray< Value, Device, Index, Communicator >::
getElement( IndexType i ) const
{
   return view.getElement( i );
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
   return view[ i ];
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
   return view[ i ];
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
   view = array;
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Array, typename..., typename >
DistributedArray< Value, Device, Index, Communicator >&
DistributedArray< Value, Device, Index, Communicator >::
operator=( const Array& array )
{
   setLike( array );
   view = array;
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
   return view == array;
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
   return view != array;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArray< Value, Device, Index, Communicator >::
containsValue( ValueType value ) const
{
   return view.containsValue( value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
bool
DistributedArray< Value, Device, Index, Communicator >::
containsOnlyValue( ValueType value ) const
{
   return view.containsOnlyValue( value );
}

} // namespace Containers
} // namespace TNL
