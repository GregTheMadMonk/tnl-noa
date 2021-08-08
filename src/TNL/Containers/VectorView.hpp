/***************************************************************************
                          VectorView_impl.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/detail/VectorAssignment.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename VectorView< Real, Device, Index >::ViewType
VectorView< Real, Device, Index >::
getView( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   return ViewType( this->getData() + begin, end - begin );;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename VectorView< Real, Device, Index >::ConstViewType
VectorView< Real, Device, Index >::
getConstView( const IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData() + begin, end - begin );;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression, typename..., typename >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< VectorView, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::modulo( *this, expression );
   return *this;
}

} // namespace Containers
} // namespace TNL
