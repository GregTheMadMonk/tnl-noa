/***************************************************************************
                          tnlSparseMatrix.h  -  description
                             -------------------
    begin                : Dec 21, 2013
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

#ifndef TNLSPARSEMATRIX_H_
#define TNLSPARSEMATRIX_H_

#include <matrices/tnlMatrix.h>
#include <matrices/tnlSparseMatrixRow.h>

template< typename Real,
          typename Device,
          typename Index >
class tnlSparseMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
   typedef typename tnlMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef tnlVector< IndexType, DeviceType, IndexType > ColumnIndexesVector;
   typedef tnlMatrix< Real, Device, Index > BaseType;
   typedef tnlSparseMatrixRow< RealType, IndexType > MatrixRow;

   tnlSparseMatrix();

   virtual bool setRowLengths( const RowLengthsVector& rowLengths ) = 0;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlSparseMatrix< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowLength() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getPaddingIndex() const;

   void reset();

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   void printStructure( ostream& str ) const;

   protected:

   bool allocateMatrixElements( const IndexType& numberOfMatrixElements );

   tnlVector< Index, Device, Index > columnIndexes;

   Index maxRowLength;
};

#include <matrices/tnlSparseMatrix_impl.h>

#endif /* TNLSPARSEMATRIX_H_ */
