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
class Vector : public Containers::Array< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Vector< Real, TNL::Devices::Host, Index > HostType;
   typedef Vector< Real, TNL::Devices::Cuda, Index > CudaType;

   Vector();

   Vector( const Index size );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void addElement( const IndexType i,
                    const RealType& value );

   void addElement( const IndexType i,
                    const RealType& value,
                    const RealType& thisElementMultiplicator );

   Vector< Real, Device, Index >& operator = ( const Vector< Real, Device, Index >& array );

   template< typename VectorT >
   Vector< Real, Device, Index >& operator = ( const VectorT& vector );

   template< typename VectorT >
   bool operator == ( const VectorT& vector ) const;

   template< typename VectorT >
   bool operator != ( const VectorT& vector ) const;

   template< typename VectorT >
   Vector< Real, Device, Index >& operator -= ( const VectorT& vector );

   template< typename VectorT >
   Vector< Real, Device, Index >& operator += ( const VectorT& vector );

   Vector< Real, Device, Index >& operator *= ( const RealType& c );
 
   Vector< Real, Device, Index >& operator /= ( const RealType& c );

   Real max() const;

   Real min() const;

   Real absMax() const;

   Real absMin() const;

   Real lpNorm( const Real& p ) const;

   Real sum() const;

   template< typename Vector >
   Real differenceMax( const Vector& v ) const;

   template< typename Vector >
   Real differenceMin( const Vector& v ) const;

   template< typename Vector >
   Real differenceAbsMax( const Vector& v ) const;

   template< typename Vector >
   Real differenceAbsMin( const Vector& v ) const;

   template< typename Vector >
   Real differenceLpNorm( const Vector& v, const Real& p ) const;

   template< typename Vector >
   Real differenceSum( const Vector& v ) const;

   void scalarMultiplication( const Real& alpha );

   //! Computes scalar dot product
   template< typename Vector >
   Real scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + multiplicator * v.
   template< typename Vector >
   void addVector( const Vector& v,
                   const Real& multiplicator = 1.0,
                   const Real& thisMultiplicator = 1.0 );


   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector >
   void addVectors( const Vector& v1,
                    const Real& multiplicator1,
                    const Vector& v2,
                    const Real& multiplicator2,
                    const Real& thisMultiplicator = 1.0 );

   void computePrefixSum();

   void computePrefixSum( const IndexType begin, const IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( const IndexType begin, const IndexType end );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Vector_impl.h>
