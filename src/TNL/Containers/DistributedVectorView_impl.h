/***************************************************************************
                          DistributedVectorView_impl.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedVectorView.h"
#include <TNL/Containers/Algorithms/ReductionOperations.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Real, Device, Index, Communicator >::LocalVectorViewType
DistributedVectorView< Real, Device, Index, Communicator >::
getLocalVectorView()
{
   return this->getLocalArrayView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType
DistributedVectorView< Real, Device, Index, Communicator >::
getLocalVectorView() const
{
   return this->getLocalArrayView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
typename DistributedVectorView< Real, Device, Index, Communicator >::ConstLocalVectorViewType
DistributedVectorView< Real, Device, Index, Communicator >::
getConstLocalVectorView() const
{
   return this->getLocalArrayView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
__cuda_callable__
typename DistributedVectorView< Value, Device, Index, Communicator >::ViewType
DistributedVectorView< Value, Device, Index, Communicator >::
getView()
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
__cuda_callable__
typename DistributedVectorView< Value, Device, Index, Communicator >::ConstViewType
DistributedVectorView< Value, Device, Index, Communicator >::
getView() const
{
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Communicator >
__cuda_callable__
typename DistributedVectorView< Value, Device, Index, Communicator >::ConstViewType
DistributedVectorView< Value, Device, Index, Communicator >::
getConstView() const
{
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
String
DistributedVectorView< Real, Device, Index, Communicator >::
getType()
{
   return String( "Containers::DistributedVectorView< " ) +
          TNL::getType< Real >() + ", " +
          Device::getDeviceType() + ", " +
          TNL::getType< Index >() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator> >";
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedVectorView< Real, Device, Index, Communicator >::
addElement( IndexType i,
            RealType value )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const IndexType li = this->getLocalRange().getLocalIndex( i );
      LocalVectorViewType view = getLocalVectorView();
      view.addElement( li, value );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar >
void
DistributedVectorView< Real, Device, Index, Communicator >::
addElement( IndexType i,
            RealType value,
            Scalar thisElementMultiplicator )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const IndexType li = this->getLocalRange().getLocalIndex( i );
      LocalVectorViewType view = getLocalVectorView();
      view.addElement( li, value, thisElementMultiplicator );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator-=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() -= vector.getLocalVectorView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator+=( const Vector& vector )
{
   TNL_ASSERT_EQ( this->getSize(), vector.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), vector.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), vector.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() += vector.getLocalVectorView();
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator*=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() *= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Scalar >
DistributedVectorView< Real, Device, Index, Communicator >&
DistributedVectorView< Real, Device, Index, Communicator >::
operator/=( Scalar c )
{
   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView() /= c;
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename ResultType >
ResultType
DistributedVectorView< Real, Device, Index, Communicator >::
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
          typename Index,
          typename Communicator >
   template< typename Vector >
typename DistributedVectorView< Real, Device, Index, Communicator >::NonConstReal
DistributedVectorView< Real, Device, Index, Communicator >::
scalarProduct( const Vector& v ) const
{
   const auto group = this->getCommunicationGroup();
   NonConstReal result = Containers::Algorithms::ParallelReductionScalarProduct< Real, typename Vector::RealType >::initialValue();
   if( group != CommunicatorType::NullGroup ) {
      const Real localResult = getLocalVectorView().scalarProduct( v.getLocalVectorView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, group );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector, typename Scalar1, typename Scalar2 >
void
DistributedVectorView< Real, Device, Index, Communicator >::
addVector( const Vector& x,
           Scalar1 alpha,
           Scalar2 thisMultiplicator )
{
   TNL_ASSERT_EQ( this->getSize(), x.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), x.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), x.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );

   if( this->getCommunicationGroup() != CommunicatorType::NullGroup ) {
      getLocalVectorView().addVector( x.getLocalVectorView(), alpha, thisMultiplicator );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 >
void
DistributedVectorView< Real, Device, Index, Communicator >::
addVectors( const Vector1& v1,
            Scalar1 multiplicator1,
            const Vector2& v2,
            Scalar2 multiplicator2,
            Scalar3 thisMultiplicator )
{
   TNL_ASSERT_EQ( this->getSize(), v1.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), v1.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), v1.getCommunicationGroup(),
                  "Multiary operations are supported only on vectors within the same communication group." );
   TNL_ASSERT_EQ( this->getSize(), v2.getSize(),
                  "Vector sizes must be equal." );
   TNL_ASSERT_EQ( this->getLocalRange(), v2.getLocalRange(),
                  "Multiary operations are supported only on vectors which are distributed the same way." );
   TNL_ASSERT_EQ( this->getCommunicationGroup(), v2.getCommunicationGroup(),
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
          typename Index,
          typename Communicator >
void
DistributedVectorView< Real, Device, Index, Communicator >::
computePrefixSum()
{
   throw Exceptions::NotImplementedError("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedVectorView< Real, Device, Index, Communicator >::
computePrefixSum( IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedVectorView< Real, Device, Index, Communicator >::
computeExclusivePrefixSum()
{
   throw Exceptions::NotImplementedError("Distributed prefix sum is not implemented yet.");
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
DistributedVectorView< Real, Device, Index, Communicator >::
computeExclusivePrefixSum( IndexType begin, IndexType end )
{
   throw Exceptions::NotImplementedError("Distributed prefix sum is not implemented yet.");
}

} // namespace Containers
} // namespace TNL
