/***************************************************************************
                          tnlSlicedEllpackMatrix.h  -  description
                             -------------------
    begin                : Dec 8, 2013
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

#ifndef TNLSLICEDELLPACKMATRIX_H_
#define TNLSLICEDELLPACKMATRIX_H_

#include <core/vectors/tnlVector.h>

template< typename Real, typename Device = tnlHost, typename Index = int, int SliceSize = 32 >
class tnlSlicedEllpackMatrix : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlSlicedEllpackMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   template< typename Vector >
   bool setRowLengths( const Vector& rowLengths );

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlSlicedEllpackMatrix< Real2, Device2, Index2, SliceSize >& matrix );

   IndexType getNumberOfAllocatedElements() const;

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   bool addToElement( const IndexType row,
                      const IndexType column,
                      const RealType& value,
                      const RealType& thisElementMultiplicator = 1.0 );

   template< typename Vector >
   void vectorProduct( const Vector& inVector,
                       Vector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( ostream& str ) const;

   protected:

   IndexType rows, columns;

   tnlVector< Real, Device, Index > values;

   tnlVector< Index, Device, Index > columnIndexes, slicePointers, sliceRowLengths;

};

#include <implementation/matrices/tnlSlicedEllpackMatrix_impl.h>


#endif /* TNLSLICEDELLPACKMATRIX_H_ */
