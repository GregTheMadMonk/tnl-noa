/***************************************************************************
                          tnlTridiagonalMatrix.h  -  description
                             -------------------
    begin                : Nov 30, 2013
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

#ifndef TNLTRIDIAGONALMATRIX_H_
#define TNLTRIDIAGONALMATRIX_H_

#include <matrices/tnlMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlTridiagonalMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< Real, Device, Index >::RowLengthsVector RowLengthsVector;

   tnlTridiagonalMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setRowLengths( const RowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& m );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const;

   void setValue( const RealType& v );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType elements,
                const RealType& thisRowMultiplicator = 1.0 );

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;

   RealType& operator()( const IndexType row,
                         const IndexType column );

   const RealType& operator()( const IndexType row,
                               const IndexType column ) const;

   template< typename Vector >
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

#ifdef HAVE_NOT_CXX11
   template< typename Real2, typename Index2, int tileDim >
   void getTransposition( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#else
   template< typename Real2, typename Index2, int tileDim = 32 >
   void getTransposition( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );
#endif   

   template< typename Vector >
   void performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   protected:

   IndexType getElementIndex( const IndexType row,
                              const IndexType column ) const;

   tnlVector< RealType, DeviceType, IndexType > values;
};

#include <implementation/matrices/tnlTridiagonalMatrix_impl.h>


#endif /* TNLTRIDIAGONALMATRIX_H_ */
