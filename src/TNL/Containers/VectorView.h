/***************************************************************************
                          VectorView.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz et al.
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/ArrayView.h>

namespace TNL {
namespace Containers {

template< typename Real, typename Device, typename Index >
class Vector;

template< int Size, typename Real >
class StaticVector;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class VectorView
: public ArrayView< Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = VectorView< Real, Devices::Host, Index >;
   using CudaType = VectorView< Real, Devices::Cuda, Index >;

   // inherit all ArrayView's constructors
   using ArrayView< Real, Device, Index >::ArrayView;

   // Copy-assignment does deep copy, just like regular vector, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   // TODO: can it be inherited?
   VectorView< Real, Device, Index >& operator=( const VectorView& view );


   static String getType();

   virtual String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;


   // All other Vector methods follow...
   void addElement( IndexType i, RealType value );

   void addElement( IndexType i,
                    RealType value,
                    RealType thisElementMultiplicator );

   using ArrayView< Real, Device, Index >::operator==;
   using ArrayView< Real, Device, Index >::operator!=;

   template< typename Vector >
   VectorView< Real, Device, Index >& operator-=( const Vector& vector );

   template< typename Vector >
   VectorView< Real, Device, Index >& operator+=( const Vector& vector );

   VectorView< Real, Device, Index >& operator*=( RealType c );

   VectorView< Real, Device, Index >& operator/=( RealType c );

   Real max() const;

   Real min() const;

   Real absMax() const;

   Real absMin() const;

   template< typename ResultType = RealType, typename Real_ >
   ResultType lpNorm( Real_ p ) const;

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
   ResultType differenceLpNorm( const Vector& v, Real_ p ) const;

   template< typename ResultType = RealType, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   void scalarMultiplication( Real alpha );

   //! Computes scalar dot product
   template< typename Vector >
   Real scalarProduct( const Vector& v );

   //! Computes Y = alpha * X + Y.
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

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/VectorView_impl.h>
