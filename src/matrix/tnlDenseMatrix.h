/***************************************************************************
                          tnlDenseMatrix.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLDENSEMATRIX_H_
#define TNLDENSEMATRIX_H_

#include <core/tnlHost.h>
#include <core/arrays/tnlMultiArray.h>

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDenseMatrix : public tnlMultiArray< 2, Real, Device, Index >
{
   public:

   typedef Real RealType;

   tnlDenseMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   template< typename Vector >
   RealType rowVectorProduct( const Vector& v ) const;

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

   template< typename Matrix1, typename Matrix2 >
   void matrixProduct( const Matrix1& m1,
                       const Matrix2& m2 );

   template< typename Matrix >
   void matrixTransposition( const Matrix& m );

   template< typename Vector >
   void performSORIteration( const Vector& b,
                             Vector& x ) const;


};


#endif /* TNLDENSEMATRIX_H_ */
