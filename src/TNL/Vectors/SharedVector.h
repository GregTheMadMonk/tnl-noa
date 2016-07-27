/***************************************************************************
                          SharedVector.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Arrays/SharedArray.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/functions/tnlDomain.h>

namespace TNL {

   class tnlHost;

namespace Vectors {   



template< typename Real = double,
           typename Device= tnlHost,
           typename Index = int >
class SharedVector : public Arrays::tnlSharedArray< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef SharedVector< Real, tnlHost, Index > HostType;
   typedef SharedVector< Real, tnlCuda, Index > CudaType;


   __cuda_callable__
   SharedVector();

   __cuda_callable__
   SharedVector( Real* data,
                    const Index size );

   __cuda_callable__
   SharedVector( Vector< Real, Device, Index >& vector );

   __cuda_callable__
   SharedVector( SharedVector< Real, Device, Index >& vector );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void addElement( const IndexType i,
                    const RealType& value );

   void addElement( const IndexType i,
                    const RealType& value,
                    const RealType& thisElementMultiplicator );

   SharedVector< Real, Device, Index >& operator = ( const SharedVector< Real, Device, Index >& array );

   template< typename Vector >
   SharedVector< Real, Device, Index >& operator = ( const Vector& array );

   template< typename Vector >
   bool operator == ( const Vector& array ) const;

   template< typename Vector >
   bool operator != ( const Vector& array ) const;

   template< typename Vector >
   SharedVector< Real, Device, Index >& operator -= ( const Vector& vector );

   template< typename Vector >
   SharedVector< Real, Device, Index >& operator += ( const Vector& vector );
 
   SharedVector< Real, Device, Index >& operator *= ( const RealType& c );
 
   SharedVector< Real, Device, Index >& operator /= ( const RealType& c );

   //bool save( File& file ) const;

   //bool save( const String& fileName ) const;

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
   Real scalarProduct( const Vector& v );

   //! Computes Y = alpha * X + Y.
   template< typename Vector >
   void addVector( const Vector& x,
                   const Real& alpha = 1.0,
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

} // namespace Vectors
} // namespace TNL

#include <TNL/Vectors/SharedVector_impl.h>

