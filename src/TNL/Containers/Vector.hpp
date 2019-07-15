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
          typename Index >
String
Vector< Real, Device, Index >::
getType()
{
   return String( "Containers::Vector< " ) +
                  TNL::getType< Real >() + ", " +
                  Device::getDeviceType() + ", " +
                  TNL::getType< Index >() + " >";
}

template< typename Real,
          typename Device,
          typename Index >
String
Vector< Real, Device, Index >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
typename Vector< Real, Device, Index >::ViewType
Vector< Real, Device, Index >::
getView( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();
   return ViewType( this->getData() + begin, end - begin );
}

template< typename Real,
          typename Device,
          typename Index >
typename Vector< Real, Device, Index >::ConstViewType
Vector< Real, Device, Index >::
getView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( &this->getData()[ begin ], end - begin );
}

template< typename Real,
          typename Device,
          typename Index >
typename Vector< Real, Device, Index >::ConstViewType
Vector< Real, Device, Index >::
getConstView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData() + begin, end - begin );
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
operator ViewType()
{
   return getView();
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Real,
          typename Device,
          typename Index >
void
Vector< Real, Device, Index >::
addElement( const IndexType i,
            const RealType& value )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar >
void
Vector< Real, Device, Index >::
addElement( const IndexType i,
            const RealType& value,
            const Scalar thisElementMultiplicator )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator=( const VectorExpression& expression )
{
   Algorithms::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator=( const Vector< Real_, Device_, Index_ >& vector )
{
   Array< Real, Device, Index >::operator=( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator=( const VectorView< Real_, Device_, Index_ >& view )
{
   Array< Real, Device, Index >::operator=( view );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator-=( const VectorExpression& expression )
{
   Algorithms::VectorSubtraction< Vector, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator+=( const VectorExpression& expression )
{
   Algorithms::VectorAddition< Vector, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator*=( const VectorExpression& expression )
{
   Algorithms::VectorMultiplication< Vector, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator/=( const VectorExpression& expression )
{
   Algorithms::VectorDivision< Vector, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector_ >
Real Vector< Real, Device, Index >::
operator,( const Vector_& v ) const
{
   static_assert( std::is_same< DeviceType, typename Vector_::DeviceType >::value, "Cannot compute product of vectors allocated on different devices." );
   return Algorithms::VectorOperations< Device >::getScalarProduct( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType >
ResultType Vector< Real, Device, Index >::sum() const
{
   return Algorithms::VectorOperations< Device >::template getVectorSum< Vector, ResultType >( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Real Vector< Real, Device, Index >::scalarProduct( const VectorT& v ) const
{
   return dot( this->getView(), v.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT, typename Scalar1, typename Scalar2 >
void
Vector< Real, Device, Index >::
addVector( const VectorT& x,
           const Scalar1 multiplicator,
           const Scalar2 thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVector( *this, x, multiplicator, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 >
void
Vector< Real, Device, Index >::
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
          typename Index >
   template< Algorithms::PrefixSumType Type >
void
Vector< Real, Device, Index >::
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
Vector< Real, Device, Index >::
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
Vector< Real, Device, Index >::
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
Vector< Real, Device, Index >::
segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError( "Prefix sum with vector expressions is not implemented." );
}

} // namespace Containers
} // namespace TNL
