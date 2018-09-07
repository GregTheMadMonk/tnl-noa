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

#include <stdexcept>  // std::runtime_error

#include "DistributedVector.h"
#include <TNL/Containers/Algorithms/ReductionOperations.h>

namespace TNL {
namespace DistributedContainers {

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
typename DistributedVector< Real, Device, Communicator, Index, IndexMap >::LocalVectorViewType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
getLocalVectorView()
{
   return this->getLocalArrayView();
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
typename DistributedVector< Real, Device, Communicator, Index, IndexMap >::ConstLocalVectorViewType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
getLocalVectorView() const
{
   return this->getLocalArrayView();
}


template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
String
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
getType()
{
   return String( "DistributedContainers::DistributedVector< " ) +
          TNL::getType< Real >() + ", " +
          Device::getDeviceType() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator>, " +
          TNL::getType< Index >() + ", " +
          IndexMap::getType() + " >";
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
String
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
getTypeVirtual() const
{
   return getType();
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
addElement( IndexType i,
            RealType value )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const IndexType li = this->getIndexMap().getLocalIndex( i );
      LocalVectorViewType view = getLocalVectorView();
      view.addElement( li, value );
   }
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
addElement( IndexType i,
            RealType value,
            RealType thisElementMultiplicator )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const IndexType li = this->getIndexMap().getLocalIndex( i );
      LocalVectorViewType view = getLocalVectorView();
      view.addElement( li, value, thisElementMultiplicator );
   }
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
DistributedVector< Real, Device, Communicator, Index, IndexMap >&
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
operator-=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getIndexMap(), vector.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), vector.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() -= vector.getLocalVectorView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
DistributedVector< Real, Device, Communicator, Index, IndexMap >&
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
operator+=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getIndexMap(), vector.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), vector.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() += vector.getLocalVectorView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
DistributedVector< Real, Device, Communicator, Index, IndexMap >&
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
operator*=( RealType c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() *= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
DistributedVector< Real, Device, Communicator, Index, IndexMap >&
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
operator/=( RealType c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() /= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
max() const
{
   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionMax< Real >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().max();
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
min() const
{
   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionMin< Real >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().min();
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
absMax() const
{
   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionAbsMax< Real >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().absMax();
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
absMin() const
{
   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionAbsMin< Real >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().absMin();
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename ResultType, typename Real_ >
ResultType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
lpNorm( const Real_ p ) const
{
   const auto group = this->getCommunicationGroup();
   ResultType result = Containers::Algorithms::ParallelReductionLpNorm< Real, ResultType, Real_ >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const ResultType localResult = std::pow( getLocalVectorView().lpNorm( p ), p );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
      result = std::pow( result, 1.0 / p );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename ResultType >
ResultType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
sum() const
{
   const auto group = this->getCommunicationGroup();
   ResultType result = Containers::Algorithms::ParallelReductionSum< Real, ResultType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const ResultType localResult = getLocalVectorView().sum();
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceMax( const Vector& v ) const
{
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionDiffMax< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().differenceMax( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceMin( const Vector& v ) const
{
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionDiffMin< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().differenceMin( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceAbsMax( const Vector& v ) const
{
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionDiffAbsMax< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().differenceAbsMax( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceAbsMin( const Vector& v ) const
{
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionDiffAbsMin< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().differenceAbsMin( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename ResultType, typename Vector, typename Real_ >
ResultType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceLpNorm( const Vector& v, const Real_ p ) const
{
   const auto group = this->getCommunicationGroup();
   ResultType result = Containers::Algorithms::ParallelReductionDiffLpNorm< Real, typename Vector::RealType, ResultType, Real_ >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const ResultType localResult = std::pow( getLocalVectorView().differenceLpNorm( v.getLocalVectorView(), p ), p );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
      result = std::pow( result, 1.0 / p );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename ResultType, typename Vector >
ResultType
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
differenceSum( const Vector& v ) const
{
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   const auto group = this->getCommunicationGroup();
   ResultType result = Containers::Algorithms::ParallelReductionDiffSum< Real, typename Vector::RealType, ResultType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const ResultType localResult = getLocalVectorView().differenceSum( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
scalarMultiplication( Real alpha )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView().scalarMultiplication( alpha );
   }
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
Real
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
scalarProduct( const Vector& v ) const
{
   const auto group = this->getCommunicationGroup();
   Real result = Containers::Algorithms::ParallelReductionScalarProduct< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().scalarProduct( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
addVector( const Vector& x,
           Real alpha,
           Real thisMultiplicator )
{
   TNL_ASSERT_EQ( this->getIndexMap(), x.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), x.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView().addVector( x.getLocalVectorView(), alpha, thisMultiplicator );
   }
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
   template< typename Vector >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
addVectors( const Vector& v1,
            Real multiplicator1,
            const Vector& v2,
            Real multiplicator2,
            Real thisMultiplicator )
{
   TNL_ASSERT_EQ( this->getIndexMap(), v1.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v1.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );
   TNL_ASSERT_EQ( this->getIndexMap(), v2.getIndexMap(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getIndexMap(), v2.getIndexMap(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView().addVectors( v1.getLocalVectorView(),
                                       multiplicator1,
                                       v2.getLocalVectorView(),
                                       multiplicator2,
                                       thisMultiplicator );
   }
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
computePrefixSum()
{
   throw std::runtime_error("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
computePrefixSum( IndexType begin, IndexType end )
{
   throw std::runtime_error("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
computeExclusivePrefixSum()
{
   throw std::runtime_error("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Communicator,
          typename Index,
          typename IndexMap >
void
DistributedVector< Real, Device, Communicator, Index, IndexMap >::
computeExclusivePrefixSum( IndexType begin, IndexType end )
{
   throw std::runtime_error("Distributed prefix sum is not implemented yet.");
}

} // namespace DistributedContainers
} // namespace TNL
