/***************************************************************************
                          Vector.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
Vector< Real, Device, Index, Allocator >::
Vector( const Vector& vector,
        const AllocatorType& allocator )
: Array< Real, Device, Index, Allocator >( vector, allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ViewType
Vector< Real, Device, Index, Allocator >::
getView( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   return ViewType( this->getData() + begin, end - begin );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ConstViewType
Vector< Real, Device, Index, Allocator >::
getConstView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData() + begin, end - begin );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
Vector< Real, Device, Index, Allocator >::
operator ViewType()
{
   return getView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
Vector< Real, Device, Index, Allocator >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorExpression, typename..., typename >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::
operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   detail::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::
operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::
operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::
operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::
operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::ScanType Type >
void
Vector< Real, Device, Index, Allocator >::
prefixSum( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::Scan< DeviceType, Type >::perform( *this, begin, end, std::plus<>{}, (RealType) 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::ScanType Type,
             typename FlagsArray >
void
Vector< Real, Device, Index, Allocator >::
segmentedPrefixSum( FlagsArray& flags, IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::SegmentedScan< DeviceType, Type >::perform( *this, flags, begin, end, std::plus<>{}, (RealType) 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::ScanType Type,
             typename VectorExpression >
void
Vector< Real, Device, Index, Allocator >::
prefixSum( const VectorExpression& expression, IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError( "Prefix sum with vector expressions is not implemented." );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::ScanType Type,
             typename VectorExpression,
             typename FlagsArray >
void
Vector< Real, Device, Index, Allocator >::
segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError( "Prefix sum with vector expressions is not implemented." );
}

} // namespace Containers
} // namespace TNL
