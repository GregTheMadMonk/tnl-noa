/***************************************************************************
                          VectorView_impl.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz et al.
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
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::
operator=( const VectorView& view )
{
   ArrayView< Real, Device, Index >::operator=( view );
   return *this;
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
String
VectorView< Real, Device, Index >::
getTypeVirtual() const
{
   return getType();
}

template< typename Real,
          typename Device,
          typename Index >
String
VectorView< Real, Device, Index >::
getSerializationType()
{
   return Vector< Real, Devices::Host, Index >::getType();
}

template< typename Real,
          typename Device,
          typename Index >
String
VectorView< Real, Device, Index >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
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
void
VectorView< Real, Device, Index >::
addElement( IndexType i, RealType value, RealType thisElementMultiplicator )
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
Real
VectorView< Real, Device, Index >::
max() const
{
   return Algorithms::VectorOperations< Device >::getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real
VectorView< Real, Device, Index >::
min() const
{
   return Algorithms::VectorOperations< Device >::getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real
VectorView< Real, Device, Index >::
absMax() const
{
   return Algorithms::VectorOperations< Device >::getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real
VectorView< Real, Device, Index >::
absMin() const
{
   return Algorithms::VectorOperations< Device >::getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real
VectorView< Real, Device, Index >::
lpNorm( Real p ) const
{
   return Algorithms::VectorOperations< Device >::getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real
VectorView< Real, Device, Index >::
sum() const
{
   return Algorithms::VectorOperations< Device >::getVectorSum( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
Real
VectorView< Real, Device, Index >::
differenceMax( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
Real
VectorView< Real, Device, Index >::differenceMin( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
Real
VectorView< Real, Device, Index >::
differenceAbsMax( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real
VectorView< Real, Device, Index >::
differenceAbsMin( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real
VectorView< Real, Device, Index >::
differenceLpNorm( const Vector& v, Real p ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceLpNorm( *this, v, p );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real
VectorView< Real, Device, Index >::
differenceSum( const Vector& v ) const
{
   return Algorithms::VectorOperations< Device >::getVectorDifferenceSum( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
void
VectorView< Real, Device, Index >::
scalarMultiplication( Real alpha )
{
   Algorithms::VectorOperations< Device >::vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real
VectorView< Real, Device, Index >::
scalarProduct( const Vector& v )
{
   return Algorithms::VectorOperations< Device >::getScalarProduct( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void
VectorView< Real, Device, Index >::
addVector( const Vector& x, Real alpha, Real thisMultiplicator )
{
   Algorithms::VectorOperations< Device >::addVector( *this, x, alpha, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void
VectorView< Real, Device, Index >::
addVectors( const Vector& v1,
            Real multiplicator1,
            const Vector& v2,
            Real multiplicator2,
            Real thisMultiplicator )
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
