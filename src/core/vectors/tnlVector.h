/***************************************************************************
                          tnlVector.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLVECTOR_H_
#define TNLVECTOR_H_

#include <core/arrays/tnlArray.h>
#include <functions/tnlFunctionType.h>

class tnlHost;

template< typename Real = double,
           typename Device = tnlHost,
           typename Index = int >
class tnlVector : public tnlArray< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< Real, tnlHost, Index > HostType;
   typedef tnlVector< Real, tnlCuda, Index > CudaType;


   tnlVector();

   tnlVector( const tnlString& name );

   tnlVector( const tnlString& name, const Index size );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void addElement( const IndexType i,
                    const RealType& value );

   void addElement( const IndexType i,
                    const RealType& value,
                    const RealType& thisElementMultiplicator );

   tnlVector< Real, Device, Index >& operator = ( const tnlVector< Real, Device, Index >& array );

   template< typename Vector >
   tnlVector< Real, Device, Index >& operator = ( const Vector& vector );

   template< typename Vector >
   bool operator == ( const Vector& vector ) const;

   template< typename Vector >
   bool operator != ( const Vector& vector ) const;

   template< typename Vector >
   tnlVector< Real, Device, Index >& operator -= ( const Vector& vector );

   template< typename Vector >
   tnlVector< Real, Device, Index >& operator += ( const Vector& vector );

   // TODO: implement
   //tnlVector< Real, Device, Index >& operator *= ( const RealType& c );

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

template< typename Real,
          typename Device,
          typename Index >
class tnlFunctionType< tnlVector< Real, Device, Index > >
{
   public:

      enum { Type = tnlDiscreteFunction };
};

#include <core/vectors/tnlVector_impl.h>

#endif /* TNLVECTOR_H_ */
