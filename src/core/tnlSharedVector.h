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

#include <core/tnlSharedArray.h>

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

   tnlString getType() const;

   tnlSharedVector< Real, Device, Index >& operator = ( const tnlSharedVector< Real, Device, Index >& array );

   template< typename Vector >
   tnlSharedVector< Real, Device, Index >& operator = ( const Vector& array );

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

   //! Compute scalar dot product
   template< typename Vector >
   Real sdot( const Vector& v );

   //! Compute SAXPY operation (Scalar Alpha X Pus Y ).
   template< typename Vector >
   void saxpy( const Real& alpha,
               const Vector& x );

   //! Compute SAXMY operation (Scalar Alpha X Minus Y ).
   /*!**
    * It is not a standart BLAS function but is useful for GMRES solver.
    */
   template< typename Vector >
   void saxmy( const Real& alpha,
               const Vector& x );
};

#include <core/implementation/tnlSharedVector_impl.h>

#endif /* TNLSHAREDVECTOR_H_ */
