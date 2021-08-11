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

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index >
typename DistributedVectorView< Real, Device, Index >::LocalViewType
DistributedVectorView< Real, Device, Index >::
getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real,
          typename Device,
          typename Index >
typename DistributedVectorView< Real, Device, Index >::ConstLocalViewType
DistributedVectorView< Real, Device, Index >::
getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Real,
          typename Device,
          typename Index >
typename DistributedVectorView< Real, Device, Index >::LocalViewType
DistributedVectorView< Real, Device, Index >::
getLocalViewWithGhosts()
{
   return BaseType::getLocalViewWithGhosts();
}

template< typename Real,
          typename Device,
          typename Index >
typename DistributedVectorView< Real, Device, Index >::ConstLocalViewType
DistributedVectorView< Real, Device, Index >::
getConstLocalViewWithGhosts() const
{
   return BaseType::getConstLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index >
typename DistributedVectorView< Value, Device, Index >::ViewType
DistributedVectorView< Value, Device, Index >::
getView()
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index >
typename DistributedVectorView< Value, Device, Index >::ConstViewType
DistributedVectorView< Value, Device, Index >::
getConstView() const
{
   return *this;
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "The sizes of the array views must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "The communication groups of the array views must be equal." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() = vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator+=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() += vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator-=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() -= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator*=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() *= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator/=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() /= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator%=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getGhosts(), vector.getGhosts(),
                  "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() %= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() = c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator+=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() += c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator-=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() -= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator*=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() *= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator/=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() /= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::
operator%=( Scalar c )
{
   if( this->getCommunicationGroup() != MPI::NullGroup() ) {
      getLocalView() %= c;
      this->startSynchronization();
   }
   return *this;
}

} // namespace Containers
} // namespace TNL
