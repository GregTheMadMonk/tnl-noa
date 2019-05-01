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
#include <TNL/Containers/Algorithms/VectorOperations.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector()
{
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( const IndexType& size )
:  Array< Real, Device, Index >( size )
{
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( Real* data,
       const IndexType& size )
:  Array< Real, Device, Index >( data, size )
{
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( const Vector< Real, Device, Index >& vector )
:  Array< Real, Device, Index >( vector )
{
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
Vector< Real, Device, Index >::
Vector( const Vector< Real_, Device_, Index_ >& vector )
:  Array< Real, Device, Index >( vector )
{
}


template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( Vector< Real, Device, Index >& vector,
        const IndexType& begin,
        const IndexType& size )
: Array< Real, Device, Index >( vector, begin, size )
{
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( Vector< Real, Device, Index >&& vector )
:  Array< Real, Device, Index >( std::move( vector ) )
{
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::
Vector( const std::initializer_list< Real >& list )
:  Array< Real, Device, Index >( list )
{
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InReal >
Vector< Real, Device, Index >::
Vector( const std::list< InReal >& list )
:  Array< Real, Device, Index >( list )
{
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InReal >
Vector< Real, Device, Index >::
Vector( const std::vector< InReal >& vector )
:   Array< Real, Device, Index >( vector )
{
}

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
getView()
{
   return ViewType( this->getData(), this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
typename Vector< Real, Device, Index >::ConstViewType
Vector< Real, Device, Index >::
getConstView() const
{
   return ConstViewType( this->getData(), this->getSize() );
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
__cuda_callable__
inline Real&
Vector< Real, Device, Index >::
operator[]( const Index& i )
{
   return Array< Real, Device, Index >::operator[]( i );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
inline const Real&
Vector< Real, Device, Index >::
operator[]( const Index& i ) const
{
   return Array< Real, Device, Index >::operator[]( i );
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator = ( const Vector& vector )
{
   Array< Real, Device, Index >::operator=( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorExpression >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator = ( const VectorExpression& expression )
{
   Algorithms::VectorAssignment< Vector< Real, Device, Index >, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator -= ( const VectorT& vector )
{
   this->addVector( vector, -1.0 );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator = ( const Vector< Real_, Device_, Index_ >& vector )
{
   Array< Real, Device, Index >::operator=( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real_, typename Device_, typename Index_ >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::operator = ( const VectorView< Real_, Device_, Index_ >& vector )
{
   Array< Real, Device, Index >::operator=( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator += ( const VectorT& vector )
{
   this->addVector( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator *= ( const Scalar c )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar >
Vector< Real, Device, Index >&
Vector< Real, Device, Index >::
operator /= ( const Scalar c )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, 1.0 / c );
   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::max() const
{
   return Algorithms::VectorOperations< Device >::getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::min() const
{
   return Algorithms::VectorOperations< Device >::getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::absMax() const
{
   return Algorithms::VectorOperations< Device >::getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::absMin() const
{
   return Algorithms::VectorOperations< Device >::getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename Scalar >
ResultType Vector< Real, Device, Index >::lpNorm( const Scalar p ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorLpNorm< Vector, ResultType >( *this, p );
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
Real Vector< Real, Device, Index >::differenceMax( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Real Vector< Real, Device, Index >::differenceMin( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Real Vector< Real, Device, Index >::differenceAbsMax( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Real Vector< Real, Device, Index >::differenceAbsMin( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename VectorT, typename Scalar >
ResultType Vector< Real, Device, Index >::differenceLpNorm( const VectorT& v, const Scalar p ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceLpNorm< Vector, VectorT, ResultType >( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename VectorT >
ResultType Vector< Real, Device, Index >::differenceSum( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceSum< Vector, VectorT, ResultType >( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Scalar >
void Vector< Real, Device, Index >::scalarMultiplication( const Scalar alpha )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Real Vector< Real, Device, Index >::scalarProduct( const VectorT& v ) const
{
   return Algorithms::VectorOperations< Device >::getScalarProduct( *this, v );
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
void Vector< Real, Device, Index >::computePrefixSum()
{
   Algorithms::VectorOperations< Device >::computePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void
Vector< Real, Device, Index >::
computePrefixSum( const IndexType begin,
                  const IndexType end )
{
   Algorithms::VectorOperations< Device >::computePrefixSum( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::computeExclusivePrefixSum()
{
   Algorithms::VectorOperations< Device >::computeExclusivePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void
Vector< Real, Device, Index >::
computeExclusivePrefixSum( const IndexType begin,
                           const IndexType end )
{
   Algorithms::VectorOperations< Device >::computeExclusivePrefixSum( *this, begin, end );
}

} // namespace Containers
} // namespace TNL
