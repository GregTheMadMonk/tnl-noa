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

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedVectorView.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedVector
: public DistributedArray< Real, Device, Index, Communicator >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using BaseType = DistributedArray< Real, Device, Index, Communicator >;
public:
   using RealType = Real;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using LocalViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalViewType = Containers::VectorView< std::add_const_t< Real >, Device, Index >;
   using HostType = DistributedVector< Real, Devices::Host, Index, Communicator >;
   using CudaType = DistributedVector< Real, Devices::Cuda, Index, Communicator >;
   using ViewType = DistributedVectorView< Real, Device, Index, Communicator >;
   using ConstViewType = DistributedVectorView< std::add_const_t< Real >, Device, Index, Communicator >;

   // inherit all constructors and assignment operators from Array
   using BaseType::DistributedArray;
   using BaseType::operator=;

   // we return only the view so that the user cannot resize it
   LocalViewType getLocalView();

   ConstLocalViewType getConstLocalView() const;

   /**
    * \brief Returns a modifiable view of the vector.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the vector.
    */
   ConstViewType getConstView() const;

   /**
    * \brief Conversion operator to a modifiable view of the vector.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the vector.
    */
   operator ConstViewType() const;


   static String getType();

   virtual String getTypeVirtual() const;


   /*
    * Usual Vector methods follow below.
    */
   void addElement( IndexType i,
                    RealType value );

   template< typename Scalar >
   void addElement( IndexType i,
                    RealType value,
                    Scalar thisElementMultiplicator );

   template< typename Vector >
   DistributedVector& operator-=( const Vector& vector );

   template< typename Vector >
   DistributedVector& operator+=( const Vector& vector );

   template< typename Scalar >
   DistributedVector& operator*=( Scalar c );

   template< typename Scalar >
   DistributedVector& operator/=( Scalar c );

   template< typename ResultType = RealType >
   ResultType sum() const;

   //! Computes scalar dot product
   template< typename Vector >
   Real scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + alpha * x.
   template< typename Vector, typename Scalar1 = Real, typename Scalar2 = Real >
   void addVector( const Vector& x,
                   Scalar1 alpha = 1.0,
                   Scalar2 thisMultiplicator = 1.0 );

   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   void addVectors( const Vector1& v1,
                    Scalar1 multiplicator1,
                    const Vector2& v2,
                    Scalar2 multiplicator2,
                    Scalar3 thisMultiplicator = 1.0 );

   void computePrefixSum();

   void computePrefixSum( IndexType begin, IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( IndexType begin, IndexType end );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/DistributedVector.hpp>
#include <TNL/Containers/DistributedVectorExpressions.h>
