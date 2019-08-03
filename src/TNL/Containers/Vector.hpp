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
String
Vector< Real, Device, Index, Allocator >::
getType()
{
   return String( "Containers::Vector< " ) +
                  TNL::getType< Real >() + ", " +
                  Device::getDeviceType() + ", " +
                  TNL::getType< Index >() + " >";
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
String
Vector< Real, Device, Index, Allocator >::
getTypeVirtual() const
{
   return this->getType();
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
getView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( &this->getData()[ begin ], end - begin );
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
void
Vector< Real, Device, Index, Allocator >::
addElement( const IndexType i,
            const RealType& value )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar >
void
Vector< Real, Device, Index, Allocator >::
addElement( const IndexType i,
            const RealType& value,
            const Scalar thisElementMultiplicator )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
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
   Algorithms::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   Algorithms::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< Vector, VectorExpression >::subtraction( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< Vector, VectorExpression >::addition( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< Vector, VectorExpression >::multiplication( *this, expression );
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
   Algorithms::VectorAssignmentWithOperation< Vector, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename ResultType >
ResultType Vector< Real, Device, Index, Allocator >::sum() const
{
   return Algorithms::VectorOperations< Device >::template getVectorSum< Vector, ResultType >( *this );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorT >
Real Vector< Real, Device, Index, Allocator >::scalarProduct( const VectorT& v ) const
{
   return dot( this->getView(), v.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename VectorT, typename Scalar1, typename Scalar2 >
void
Vector< Real, Device, Index, Allocator >::
addVector( const VectorT& x,
           const Scalar1 multiplicator,
           const Scalar2 thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVector( *this, x, multiplicator, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 >
void
Vector< Real, Device, Index, Allocator >::
addVectors( const Vector1& v1,
            const Scalar1 multiplicator1,
            const Vector2& v2,
            const Scalar2 multiplicator2,
            const Scalar3 thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVectors( *this, v1, multiplicator1, v2, multiplicator2, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::PrefixSumType Type >
void
Vector< Real, Device, Index, Allocator >::
prefixSum( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::VectorOperations< Device >::template prefixSum< Type >( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::PrefixSumType Type,
             typename FlagsArray >
void
Vector< Real, Device, Index, Allocator >::
segmentedPrefixSum( FlagsArray& flags, IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   Algorithms::VectorOperations< Device >::template segmentedPrefixSum< Type >( *this, flags, begin, end );
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< Algorithms::PrefixSumType Type,
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
   template< Algorithms::PrefixSumType Type,
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
