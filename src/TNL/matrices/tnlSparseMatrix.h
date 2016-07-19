/***************************************************************************
                          tnlSparseMatrix.h  -  description
                             -------------------
    begin                : Dec 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/matrices/tnlMatrix.h>
#include <TNL/matrices/tnlSparseMatrixRow.h>

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
class tnlSparseMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlMatrix< RealType, DeviceType, IndexType >::CompressedRowsLengthsVector CompressedRowsLengthsVector;
   typedef typename tnlMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef tnlVector< IndexType, DeviceType, IndexType > ColumnIndexesVector;
   typedef tnlMatrix< Real, Device, Index > BaseType;
   typedef tnlSparseMatrixRow< RealType, IndexType > MatrixRow;

   tnlSparseMatrix();

   virtual bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths ) = 0;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlSparseMatrix< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowLength() const;

   __cuda_callable__
   IndexType getPaddingIndex() const;

   void reset();

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   void printStructure( std::ostream& str ) const;

   protected:

   bool allocateMatrixElements( const IndexType& numberOfMatrixElements );

   tnlVector< Index, Device, Index > columnIndexes;

   Index maxRowLength;
};

} // namespace TNL

#include <TNL/matrices/tnlSparseMatrix_impl.h>
