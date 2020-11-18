/***************************************************************************
                          DistributedVector_impl.h  -  description
                             -------------------
    begin                : Sep 7, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedVector.h"
#include <TNL/Algorithms/DistributedScan.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVector< Real, Device, Index, Communicator >::LocalViewType
DistributedVector< Real, Device, Index, Communicator >::
getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVector< Real, Device, Index, Communicator >::ConstLocalViewType
DistributedVector< Real, Device, Index, Communicator >::
getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVector< Value, Device, Index, Communicator >::ViewType
DistributedVector< Value, Device, Index, Communicator >::
getView()
{
   return BaseType::getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVector< Value, Device, Index, Communicator >::ConstViewType
DistributedVector< Value, Device, Index, Communicator >::
getConstView() const
{
   return BaseType::getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedVector< Value, Device, Index, Communicator >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
DistributedVector< Value, Device, Index, Communicator >::
operator ConstViewType() const
{
   return getConstView();
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator=( const Vector& vector )
{
   this->setLike( vector );
   getView() = vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator+=( const Vector& vector )
{
   getView() += vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator-=( const Vector& vector )
{
   getView() -= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator*=( const Vector& vector )
{
   getView() *= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator/=( const Vector& vector )
{
   getView() /= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator=( Scalar c )
{
   getView() = c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator+=( Scalar c )
{
   getView() += c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator-=( Scalar c )
{
   getView() -= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator*=( Scalar c )
{
   getView() *= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Communicator >&
DistributedVector< Real, Device, Index, Communicator >::
operator/=( Scalar c )
{
   getView() /= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< Algorithms::ScanType Type >
void
DistributedVector< Real, Device, Index, Communicator >::
scan( IndexType begin, IndexType end )
{
   getView().template scan< Type >( begin, end );
}

} // namespace Containers
} // namespace TNL
