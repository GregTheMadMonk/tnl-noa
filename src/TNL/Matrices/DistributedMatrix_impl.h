/***************************************************************************
                          DistributedMatrix.h  -  description
                             -------------------
    begin                : Sep 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedMatrix.h"

namespace TNL {
namespace Matrices {

template< typename Matrix,
          typename Communicator >
DistributedMatrix< Matrix, Communicator >::
DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, CommunicationGroup group )
{
   setDistribution( localRowRange, rows, columns, group );
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, CommunicationGroup group )
{
   this->localRowRange = localRowRange;
   this->rows = rows;
   this->group = group;
   if( group != Communicator::NullGroup )
      localMatrix.setDimensions( localRowRange.getSize(), columns );

   spmv.reset();
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
const Containers::Subrange< typename Matrix::IndexType >&
DistributedMatrix< Matrix, Communicator >::
getLocalRowRange() const
{
   return localRowRange;
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename Communicator::CommunicationGroup
DistributedMatrix< Matrix, Communicator >::
getCommunicationGroup() const
{
   return group;
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
const Matrix&
DistributedMatrix< Matrix, Communicator >::
getLocalMatrix() const
{
   return localMatrix;
}


/*
 * Some common Matrix methods follow below.
 */

template< typename Matrix,
          typename Communicator >
DistributedMatrix< Matrix, Communicator >&
DistributedMatrix< Matrix, Communicator >::
operator=( const DistributedMatrix& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix,
          typename Communicator >
   template< typename MatrixT >
DistributedMatrix< Matrix, Communicator >&
DistributedMatrix< Matrix, Communicator >::
operator=( const MatrixT& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix,
          typename Communicator >
   template< typename MatrixT >
void
DistributedMatrix< Matrix, Communicator >::
setLike( const MatrixT& matrix )
{
   localRowRange = matrix.getLocalRowRange();
   rows = matrix.getRows();
   group = matrix.getCommunicationGroup();
   localMatrix.setLike( matrix.getLocalMatrix() );

   spmv.reset();
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
reset()
{
   localRowRange.reset();
   rows = 0;
   group = Communicator::NullGroup;
   localMatrix.reset();

   spmv.reset();
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getRows() const
{
   return rows;
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getColumns() const
{
   return localMatrix.getColumns();
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths )
{
   TNL_ASSERT_EQ( rowLengths.getSize(), getRows(), "row lengths vector has wrong size" );
   TNL_ASSERT_EQ( rowLengths.getLocalRange(), getLocalRowRange(), "row lengths vector has wrong distribution" );
   TNL_ASSERT_EQ( rowLengths.getCommunicationGroup(), getCommunicationGroup(), "row lengths vector has wrong communication group" );

   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      localMatrix.setCompressedRowLengths( rowLengths.getConstLocalView() );

      spmv.reset();
   }
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const
{
   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      rowLengths.setDistribution( getLocalRowRange(), getRows(), getCommunicationGroup() );
      localMatrix.getCompressedRowLengths( rowLengths.getLocalView() );
   }
}

template< typename Matrix,
          typename Communicator >
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getRowLength( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRowLength( localRow );
}

template< typename Matrix,
          typename Communicator >
bool
DistributedMatrix< Matrix, Communicator >::
setElement( IndexType row,
            IndexType column,
            RealType value )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.setElement( localRow, column, value );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
bool
DistributedMatrix< Matrix, Communicator >::
setElementFast( IndexType row,
                IndexType column,
                RealType value )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.setElementFast( localRow, column, value );
}

template< typename Matrix,
          typename Communicator >
typename Matrix::RealType
DistributedMatrix< Matrix, Communicator >::
getElement( IndexType row,
            IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElement( localRow, column );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename Matrix::RealType
DistributedMatrix< Matrix, Communicator >::
getElementFast( IndexType row,
                IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElementFast( localRow, column );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
bool
DistributedMatrix< Matrix, Communicator >::
setRowFast( IndexType row,
            const IndexType* columnIndexes,
            const RealType* values,
            IndexType elements )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.setRowFast( localRow, columnIndexes, values, elements );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
void
DistributedMatrix< Matrix, Communicator >::
getRowFast( IndexType row,
            IndexType* columns,
            RealType* values ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRowFast( localRow, columns, values );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename DistributedMatrix< Matrix, Communicator >::MatrixRow
DistributedMatrix< Matrix, Communicator >::
getRow( IndexType row )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix,
          typename Communicator >
__cuda_callable__
typename DistributedMatrix< Matrix, Communicator >::ConstMatrixRow
DistributedMatrix< Matrix, Communicator >::
getRow( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix,
          typename Communicator >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< ! has_communicator< InVector >::value >::type
DistributedMatrix< Matrix, Communicator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   auto outView = outVector.getLocalView();
   localMatrix.vectorProduct( inVector, outView );
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
updateVectorProductCommunicationPattern()
{
   if( getCommunicationGroup() == CommunicatorType::NullGroup )
      return;
   spmv.updateCommunicationPattern( getLocalMatrix(), getCommunicationGroup() );
}

template< typename Matrix,
          typename Communicator >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< has_communicator< InVector >::value >::type
DistributedMatrix< Matrix, Communicator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( inVector.getLocalRange(), getLocalRowRange(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicationGroup(), getCommunicationGroup(), "input vector has wrong communication group" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   if( getCommunicationGroup() == CommunicatorType::NullGroup )
      return;

   const_cast< DistributedMatrix* >( this )->spmv.vectorProduct( outVector, localMatrix, inVector, getCommunicationGroup() );
}

} // namespace Matrices
} // namespace TNL
