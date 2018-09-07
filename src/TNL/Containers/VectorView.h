/***************************************************************************
                          VectorView.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
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
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = VectorView< Real, Devices::Host, Index >;
   using CudaType = VectorView< Real, Devices::Cuda, Index >;

   // inherit all ArrayView's constructors
   using ArrayView< Real, Device, Index >::ArrayView;


   static String getType();


   // All other Vector methods follow...
   void addElement( IndexType i, RealType value );

   void addElement( IndexType i,
                    RealType value,
                    RealType thisElementMultiplicator );

   using ArrayView< Real, Device, Index >::operator==;
   using ArrayView< Real, Device, Index >::operator!=;

   template< typename Vector >
   VectorView& operator-=( const Vector& vector );

   template< typename Vector >
   VectorView& operator+=( const Vector& vector );

   VectorView& operator*=( RealType c );

   VectorView& operator/=( RealType c );

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
   NonConstReal scalarProduct( const Vector& v );

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

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/VectorView_impl.h>
