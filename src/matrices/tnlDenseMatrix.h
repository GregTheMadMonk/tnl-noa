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
#include <matrices/tnlMatrix.h>
#include <core/arrays/tnlMultiArray.h>

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDenseMatrix : public tnlMatrix< Real, Device, Index >,
                       public tnlMultiArray< 2, Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< Real, Device, Index >::RowLengthsVector RowLengthsVector;

   tnlDenseMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlDenseMatrix< Real2, Device2, Index2 >& matrix );

   /****
    * This method is only for the compatibility with the sparse matrices.
    */
   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getNumberOfMatrixElements() const;

   void reset();

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   Real getElement( const IndexType row,
                    const IndexType column ) const;

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements,
                const RealType thisRowMultiplicator = 1.0 );

   template< typename Vector >
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

   template< typename Matrix >
   void addMatrix( const Matrix& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Matrix1, typename Matrix2, int tileDim = 32 >
   void getMatrixProduct( const Matrix1& matrix1,
                       const Matrix2& matrix2,
                       const RealType& matrix1Multiplicator = 1.0,
                       const RealType& matrix2Multiplicator = 1.0 );

   template< typename Matrix, int tileDim = 32 >
   void getTransposition( const Matrix& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   void performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   void print( ostream& str ) const;
};

#include <implementation/matrices/tnlDenseMatrix_impl.h>

#endif /* TNLDENSEMATRIX_H_ */
