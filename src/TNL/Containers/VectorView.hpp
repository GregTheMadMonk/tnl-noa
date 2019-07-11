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
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
VectorView< Real, Device, Index >::VectorView( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   Algorithms::VectorAssignment< VectorView< Real, Device, Index >, Expressions::BinaryExpressionTemplate< T1, T2, Operation > >::assign( *this, expression );
};

template< typename Real,
          typename Device,
          typename Index >
   template< typename T,
             template< typename > class Operation >
__cuda_callable__
VectorView< Real, Device, Index >::VectorView( const Expressions::UnaryExpressionTemplate< T, Operation >& expression )
{
   Algorithms::VectorAssignment< VectorView< Real, Device, Index >, Expressions::UnaryExpressionTemplate< T, Operation > >::assign( *this, expression );
};

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
getView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData()[ begin ], end - begin );;
}

template< typename Real,
          typename Device,
          typename Index >
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
void
VectorView< Real, Device, Index >::
addElement( IndexType i, RealType value )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar >
void
VectorView< Real, Device, Index >::
addElement( IndexType i, RealType value, Scalar thisElementMultiplicator )
{
   Algorithms::VectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator=( const VectorView< Real_, Device_, Index_ >& v )
{
   ArrayView< Real, Device, Index >::operator=( v );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
  template< typename Real_, typename Device_, typename Index_ >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator=( const Vector< Real_, Device_, Index_ >& v )
{
   ArrayView< Real, Device, Index >::operator=( v );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
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
   Algorithms::VectorSubtraction< VectorView, VectorExpression >::subtraction( *this, expression );
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
   Algorithms::VectorAddition< VectorView, VectorExpression >::addition( *this, expression );
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
   Algorithms::VectorMultiplication< VectorView, VectorExpression >::multiplication( *this, expression );
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
   Algorithms::VectorDivision< VectorView, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector_ >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
operator,( const Vector_& v ) const
{
   static_assert( std::is_same< DeviceType, typename Vector_::DeviceType >::value, "Cannot compute product of vectors allocated on different devices." );
   return Algorithms::VectorOperations< Device >::getScalarProduct( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
max() const
{
   return Algorithms::VectorOperations< Device >::template getVectorMax< VectorView, NonConstReal >( *this );
}

template< typename Real,
          typename Device,
          typename Index >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
min() const
{
   return Algorithms::VectorOperations< Device >::template getVectorMin< VectorView, NonConstReal >( *this );
}


template< typename Real,
          typename Device,
          typename Index >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
absMax() const
{
   return Algorithms::VectorOperations< Device >::template getVectorAbsMax< VectorView, NonConstReal >( *this );
}

template< typename Real,
          typename Device,
          typename Index >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
absMin() const
{
   return Algorithms::VectorOperations< Device >::template getVectorAbsMin< VectorView, NonConstReal >( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename Scalar >
ResultType
VectorView< Real, Device, Index >::
lpNorm( const Scalar p ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorLpNorm< VectorView, ResultType >( *this, p );
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
   template< typename Scalar >
void
VectorView< Real, Device, Index >::
scalarMultiplication( Scalar alpha )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
scalarProduct( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getScalarProduct< VectorView, Vector, NonConstReal >( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector, typename Scalar1, typename Scalar2 >
void
VectorView< Real, Device, Index >::
addVector( const Vector& x, Scalar1 alpha, Scalar2 thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVector( *this, x, alpha, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 >
void
VectorView< Real, Device, Index >::
addVectors( const Vector1& v1,
            Scalar1 multiplicator1,
            const Vector2& v2,
            Scalar2 multiplicator2,
            Scalar3 thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVectors( *this, v1, multiplicator1, v2, multiplicator2, thisMultiplicator );
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
