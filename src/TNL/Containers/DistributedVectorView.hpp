/***************************************************************************
                          DistributedVectorView_impl.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedVectorView.h"
#include <TNL/Algorithms/DistributedScan.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Real, Device, Index, Communicator >::LocalViewType
DistributedVectorView< Real, Device, Index, Communicator >::
getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalViewType
DistributedVectorView< Real, Device, Index, Communicator >::
getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Value, Device, Index, Communicator >::ViewType
DistributedVectorView< Value, Device, Index, Communicator >::
getView()
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Value, Device, Index, Communicator >::ConstViewType
DistributedVectorView< Value, Device, Index, Communicator >::
getConstView() const
{
   return *this;
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "The sizes of the array views must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "The communication groups of the array views must be equal." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() = vector.getConstLocalView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator+=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() += vector.getConstLocalView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator-=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() -= vector.getConstLocalView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator*=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() *= vector.getConstLocalView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator/=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() /= vector.getConstLocalView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() = c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator+=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() += c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator-=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() -= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator*=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() *= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator/=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalView() /= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< Algorithms::ScanType Type >
void
DistributedVectorView< Real, Device, Index, Communicator >::
scan( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::DistributedScan< Type >::perform( *this, begin, end, std::plus<>{}, (RealType) 0.0 );
}

} // namespace Containers
} // namespace TNL
