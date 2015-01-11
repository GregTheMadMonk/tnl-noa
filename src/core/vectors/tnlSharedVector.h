/***************************************************************************
                          tnlSharedVector.h  -  description
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

#ifndef TNLSHAREDVECTOR_H_
#define TNLSHAREDVECTOR_H_

#include <core/arrays/tnlSharedArray.h>
#include <core/vectors/tnlVector.h>
#include <functions/tnlFunctionType.h>

class tnlHost;

template< typename Real = double,
           typename Device= tnlHost,
           typename Index = int >
class tnlSharedVector : public tnlSharedArray< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< Real, tnlHost, Index > HostType;
   typedef tnlSharedVector< Real, tnlCuda, Index > CudaType;


   tnlSharedVector();

   tnlSharedVector( Real* data,
                    const Index size );

   tnlSharedVector( tnlVector< Real, Device, Index >& vector );

   tnlSharedVector( tnlSharedVector< Real, Device, Index >& vector );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void addElement( const IndexType i,
                    const RealType& value );

   void addElement( const IndexType i,
                    const RealType& value,
                    const RealType& thisElementMultiplicator );

   tnlSharedVector< Real, Device, Index >& operator = ( const tnlSharedVector< Real, Device, Index >& array );

   template< typename Vector >
   tnlSharedVector< Real, Device, Index >& operator = ( const Vector& array );

   template< typename Vector >
   bool operator == ( const Vector& array ) const;

   template< typename Vector >
   bool operator != ( const Vector& array ) const;

   template< typename Vector >
   tnlSharedVector< Real, Device, Index >& operator -= ( const Vector& vector );

   template< typename Vector >
   tnlSharedVector< Real, Device, Index >& operator += ( const Vector& vector );

   //bool save( tnlFile& file ) const;

   //bool save( const tnlString& fileName ) const;

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

   //! Computes Y = alpha * X + beta * Y.
   template< typename Vector >
   void alphaXPlusBetaY( const Real& alpha,
                         const Vector& x,
                         const Real& beta );

   //! Computes Y = alpha * X + beta * Z
   template< typename Vector >
   void alphaXPlusBetaZ( const Real& alpha,
                         const Vector& x,
                         const Real& beta,
                         const Vector& z );

   //! Computes Y = Scalar Alpha X Plus Scalar Beta Z Plus Y
   template< typename Vector >
   void alphaXPlusBetaZPlusY( const Real& alpha,
                              const Vector& x,
                              const Real& beta,
                              const Vector& z );

   void computePrefixSum();

   void computePrefixSum( const IndexType begin, const IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( const IndexType begin, const IndexType end );

};

template< typename Real,
          typename Device,
          typename Index >
class tnlFunctionType< tnlSharedVector< Real, Device, Index > >
{
   public:

      enum { Type = tnlDiscreteFunction };
};

#include <implementation/core/vectors/tnlSharedVector_impl.h>

#endif /* TNLSHAREDVECTOR_H_ */
