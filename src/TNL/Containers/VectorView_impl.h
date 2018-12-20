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

namespace TNL {
namespace Containers {

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
   template< typename Vector >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator-=( const Vector& vector )
{
   addVector( vector, -1.0 );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator+=( const Vector& vector )
{
   addVector( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator*=( RealType c )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator/=( RealType c )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, 1.0 / c );
   return *this;
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
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
differenceMax( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceMax< VectorView, Vector, NonConstReal >( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::differenceMin( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceMin< VectorView, Vector, NonConstReal >( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
differenceAbsMax( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceAbsMax< VectorView, Vector, NonConstReal >( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename VectorView< Real, Device, Index >::NonConstReal
VectorView< Real, Device, Index >::
differenceAbsMin( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceAbsMin< VectorView, Vector, NonConstReal >( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename Vector, typename Scalar >
ResultType
VectorView< Real, Device, Index >::
differenceLpNorm( const Vector& v, const Scalar p ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceLpNorm< VectorView, Vector, ResultType >( *this, v, p );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename ResultType, typename Vector >
ResultType
VectorView< Real, Device, Index >::
differenceSum( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::template getVectorDifferenceSum< VectorView, Vector, ResultType >( *this, v );
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
void
VectorView< Real, Device, Index >::
computePrefixSum()
{
   Algorithms::VectorOperations< Device >::computePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void
VectorView< Real, Device, Index >::
computePrefixSum( IndexType begin, IndexType end )
{
   Algorithms::VectorOperations< Device >::computePrefixSum( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
void
VectorView< Real, Device, Index >::
computeExclusivePrefixSum()
{
   Algorithms::VectorOperations< Device >::computeExclusivePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void
VectorView< Real, Device, Index >::
computeExclusivePrefixSum( IndexType begin, IndexType end )
{
   Algorithms::VectorOperations< Device >::computeExclusivePrefixSum( *this, begin, end );
}

} // namespace Containers
} // namespace TNL
