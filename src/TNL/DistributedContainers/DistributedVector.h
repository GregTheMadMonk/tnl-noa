/***************************************************************************
                          DistributedVector.h  -  description
                             -------------------
    begin                : Sep 7, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedArray.h"
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace DistributedContainers {

template< typename Real,
          typename Device = Devices::Host,
          typename Communicator = Communicators::MpiCommunicator,
          typename Index = int,
          typename IndexMap = Subrange< Index > >
class DistributedVector
: public DistributedArray< Real, Device, Communicator, Index, IndexMap >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using BaseType = DistributedArray< Real, Device, Communicator, Index, IndexMap >;
public:
   using RealType = Real;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using IndexMapType = IndexMap;
   using LocalVectorViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalVectorViewType = Containers::VectorView< typename std::add_const< Real >::type, Device, Index >;
   using HostType = DistributedVector< Real, Devices::Host, Communicator, Index, IndexMap >;
   using CudaType = DistributedVector< Real, Devices::Cuda, Communicator, Index, IndexMap >;

   // inherit all constructors and assignment operators from Array
   using BaseType::DistributedArray;
   using BaseType::operator=;

   // we return only the view so that the user cannot resize it
   LocalVectorViewType getLocalVectorView();

   ConstLocalVectorViewType getLocalVectorView() const;


   static String getType();

   virtual String getTypeVirtual() const;


   /*
    * Usual Vector methods follow below.
    */
   void addElement( IndexType i,
                    RealType value );

   void addElement( IndexType i,
                    RealType value,
                    RealType thisElementMultiplicator );

   template< typename Vector >
   DistributedVector& operator-=( const Vector& vector );

   template< typename Vector >
   DistributedVector& operator+=( const Vector& vector );

   DistributedVector& operator*=( RealType c );

   DistributedVector& operator/=( RealType c );

   Real max() const;

   Real min() const;

   Real absMax() const;

   Real absMin() const;

   template< typename ResultType = RealType, typename Real_ >
   ResultType lpNorm( const Real_ p ) const;

   template< typename ResultType = RealType >
   ResultType sum() const;

   template< typename Vector >
   Real differenceMax( const Vector& v ) const;

   template< typename Vector >
   Real differenceMin( const Vector& v ) const;

   template< typename Vector >
   Real differenceAbsMax( const Vector& v ) const;

   template< typename Vector >
   Real differenceAbsMin( const Vector& v ) const;

   template< typename ResultType = RealType, typename Vector, typename Real_ >
   ResultType differenceLpNorm( const Vector& v, const Real_ p ) const;

   template< typename ResultType = RealType, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   void scalarMultiplication( Real alpha );

   //! Computes scalar dot product
   template< typename Vector >
   Real scalarProduct( const Vector& v ) const;

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

#include "DistributedVector_impl.h"
