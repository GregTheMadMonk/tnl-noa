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

#include <core/tnlArray.h>

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

   tnlVector();

   tnlVector( const tnlString& name );

   tnlVector( const tnlString& name, const Index size );

   tnlString getType() const;

   tnlVector< Real, Device, Index >& operator = ( const tnlVector< Real, Device, Index >& array );

   template< typename Vector >
   tnlVector< Real, Device, Index >& operator = ( const Vector& array );

   template< typename Vector >
   bool operator == ( const Vector& array ) const;

   template< typename Vector >
   bool operator != ( const Vector& array ) const;

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
   Real sdot( const Vector& v );

   //! Computes SAXPY operation (Y = Scalar Alpha X Plus Y ).
   template< typename Vector >
   void saxpy( const Real& alpha,
               const Vector& x );

   //! Computes SAXMY operation (Y = Scalar Alpha X Minus Y ).
   /*!**
    * It is not a standard BLAS function but is useful for linear solvers.
    */
   template< typename Vector >
   void saxmy( const Real& alpha,
               const Vector& x );

   //! Computes Y = Scalar Alpha X Plus Scalar Beta Y
   /*!**
    * It is not standard BLAS function as well.
    */
   template< typename Vector >
   void saxpsby( const Real& alpha,
                 const Vector& x,
                 const Real& beta );
};

#include <core/implementation/tnlVector_impl.h>

#endif /* TNLVECTOR_H_ */
