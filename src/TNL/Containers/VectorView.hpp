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
#include <TNL/Containers/Algorithms/VectorOperations.h>
#include <TNL/Containers/VectorViewExpressions.h>
#include <TNL/Containers/Algorithms/VectorAssignment.h>
#include <TNL/Exceptions/NotImplementedError.h>

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
String
VectorView< Real, Device, Index >::
getType()
{
   return String( "Containers::VectorView< " ) +
                  TNL::getType< Real >() + ", " +
                  Device::getDeviceType() + ", " +
                  TNL::getType< Index >() + " >";
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression, typename..., typename >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator=( const VectorExpression& expression )
{
   Algorithms::VectorAssignment< VectorView, VectorExpression >::assign( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< VectorView, VectorExpression >::subtraction( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< VectorView, VectorExpression >::addition( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< VectorView, VectorExpression >::multiplication( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< VectorView, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType >
ResultType
VectorView< Real, Device, Index >::
sum() const
{
   return Algorithms::VectorOperations< Device >::template getVectorSum< VectorView, ResultType >( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
scalarProduct( const Vector& v ) const
{
   return TNL::sum( *this * v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< Algorithms::PrefixSumType Type >
void
VectorView< Real, Device, Index >::
prefixSum( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::VectorOperations< Device >::template prefixSum< Type >( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
   template< Algorithms::PrefixSumType Type,
             typename FlagsArray >
void
VectorView< Real, Device, Index >::
segmentedPrefixSum( FlagsArray& flags, IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::VectorOperations< Device >::template segmentedPrefixSum< Type >( *this, flags, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
   template< Algorithms::PrefixSumType Type,
             typename VectorExpression >
void
VectorView< Real, Device, Index >::
prefixSum( const VectorExpression& expression, IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError( "Prefix sum with vector expressions is not implemented." );
}

template< typename Real,
          typename Device,
          typename Index >
   template< Algorithms::PrefixSumType Type,
             typename VectorExpression,
             typename FlagsArray >
void
VectorView< Real, Device, Index >::
segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError( "Prefix sum with vector expressions is not implemented." );
}

} // namespace Containers
} // namespace TNL
