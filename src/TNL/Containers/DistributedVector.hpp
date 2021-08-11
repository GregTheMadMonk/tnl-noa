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

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Real, Device, Index, Allocator >::
DistributedVector( const DistributedVector& vector, const AllocatorType& allocator )
: BaseType::DistributedArray( vector, allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::LocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::ConstLocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::LocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getLocalViewWithGhosts()
{
   return BaseType::getLocalViewWithGhosts();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::ConstLocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getConstLocalViewWithGhosts() const
{
   return BaseType::getConstLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Value, Device, Index, Allocator >::ViewType
DistributedVector< Value, Device, Index, Allocator >::
getView()
{
   return BaseType::getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Value, Device, Index, Allocator >::ConstViewType
DistributedVector< Value, Device, Index, Allocator >::
getConstView() const
{
   return BaseType::getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Value, Device, Index, Allocator >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Value, Device, Index, Allocator >::
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
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator=( const Vector& vector )
{
   this->setLike( vector );
   getView() = vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator+=( const Vector& vector )
{
   getView() += vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator-=( const Vector& vector )
{
   getView() -= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator*=( const Vector& vector )
{
   getView() *= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator/=( const Vector& vector )
{
   getView() /= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator%=( const Vector& vector )
{
   getView() %= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator=( Scalar c )
{
   getView() = c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator+=( Scalar c )
{
   getView() += c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator-=( Scalar c )
{
   getView() -= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator*=( Scalar c )
{
   getView() *= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator/=( Scalar c )
{
   getView() /= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator%=( Scalar c )
{
   getView() %= c;
   return *this;
}

} // namespace Containers
} // namespace TNL
