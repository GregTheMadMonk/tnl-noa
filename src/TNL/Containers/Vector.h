/***************************************************************************
                          Vector.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Vector
: public Array< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Vector< Real, TNL::Devices::Host, Index > HostType;
   typedef Vector< Real, TNL::Devices::Cuda, Index > CudaType;

   // inherit all constructors and assignment operators from Array
   using Array< Real, Device, Index >::Array;
   using Array< Real, Device, Index >::operator=;

   static String getType();

   virtual String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void addElement( const IndexType i,
                    const RealType& value );

   template< typename Scalar >
   void addElement( const IndexType i,
                    const RealType& value,
                    const Scalar thisElementMultiplicator );

   template< typename VectorT >
   Vector& operator -= ( const VectorT& vector );

   template< typename VectorT >
   Vector& operator += ( const VectorT& vector );

   Vector& operator *= ( const RealType& c );

   Vector& operator /= ( const RealType& c );

   Real max() const;

   Real min() const;

   Real absMax() const;

   Real absMin() const;

   template< typename ResultType = RealType, typename Scalar >
   ResultType lpNorm( const Scalar p ) const;

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

   template< typename ResultType = RealType, typename Vector, typename Scalar >
   ResultType differenceLpNorm( const Vector& v, const Scalar p ) const;

   template< typename ResultType = RealType, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   template< typename Scalar >
   void scalarMultiplication( const Scalar alpha );

   //! Computes scalar dot product
   template< typename Vector >
   Real scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + multiplicator * v.
   template< typename Vector, typename Scalar1 = Real, typename Scalar2 = Real >
   void addVector( const Vector& v,
                   const Scalar1 multiplicator = 1.0,
                   const Scalar2 thisMultiplicator = 1.0 );

   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   void addVectors( const Vector1& v1,
                    const Scalar1 multiplicator1,
                    const Vector2& v2,
                    const Scalar2 multiplicator2,
                    const Scalar3 thisMultiplicator = 1.0 );

   void computePrefixSum();

   void computePrefixSum( const IndexType begin, const IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( const IndexType begin, const IndexType end );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Vector_impl.h>
