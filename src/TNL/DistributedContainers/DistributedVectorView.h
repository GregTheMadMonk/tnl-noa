/***************************************************************************
                          DistributedVectorView.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/DistributedContainers/DistributedArrayView.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace DistributedContainers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedVectorView
: public DistributedArrayView< Real, Device, Index, Communicator >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using BaseType = DistributedArrayView< Real, Device, Index, Communicator >;
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using LocalVectorViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalVectorViewType = Containers::VectorView< typename std::add_const< Real >::type, Device, Index >;
   using HostType = DistributedVectorView< Real, Devices::Host, Index, Communicator >;
   using CudaType = DistributedVectorView< Real, Devices::Cuda, Index, Communicator >;

   // inherit all constructors and assignment operators from ArrayView
   using BaseType::DistributedArrayView;
   using BaseType::operator=;

   LocalVectorViewType getLocalVectorView();

   ConstLocalVectorViewType getLocalVectorView() const;


   static String getType();


   /*
    * Usual Vector methods follow below.
    */
   void addElement( IndexType i,
                    RealType value );

   void addElement( IndexType i,
                    RealType value,
                    RealType thisElementMultiplicator );

   template< typename Vector >
   DistributedVectorView& operator-=( const Vector& vector );

   template< typename Vector >
   DistributedVectorView& operator+=( const Vector& vector );

   DistributedVectorView& operator*=( RealType c );

   DistributedVectorView& operator/=( RealType c );

   NonConstReal max() const;

   NonConstReal min() const;

   NonConstReal absMax() const;

   NonConstReal absMin() const;

   template< typename ResultType = NonConstReal, typename Real_ >
   ResultType lpNorm( Real_ p ) const;

   template< typename ResultType = NonConstReal >
   ResultType sum() const;

   template< typename Vector >
   NonConstReal differenceMax( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceMin( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceAbsMax( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceAbsMin( const Vector& v ) const;

   template< typename ResultType = NonConstReal, typename Vector, typename Real_ >
   ResultType differenceLpNorm( const Vector& v, Real_ p ) const;

   template< typename ResultType = NonConstReal, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   void scalarMultiplication( Real alpha );

   //! Computes scalar dot product
   template< typename Vector >
   NonConstReal scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + alpha * x.
   template< typename Vector >
   void addVector( const Vector& x,
                   Real alpha = 1.0,
                   Real thisMultiplicator = 1.0 );

   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector >
   void addVectors( const Vector& v1,
                    Real multiplicator1,
                    const Vector& v2,
                    Real multiplicator2,
                    Real thisMultiplicator = 1.0 );

   void computePrefixSum();

   void computePrefixSum( IndexType begin, IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( IndexType begin, IndexType end );
};

} // namespace DistributedContainers
} // namespace TNL

#include "DistributedVectorView_impl.h"
