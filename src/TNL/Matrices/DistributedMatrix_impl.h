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
const Containers::Subrange< typename Matrix::IndexType >&
DistributedMatrix< Matrix, Communicator >::
getLocalRowRange() const
{
   return localRowRange;
}

template< typename Matrix,
          typename Communicator >
typename Communicator::CommunicationGroup
DistributedMatrix< Matrix, Communicator >::
getCommunicationGroup() const
{
   return group;
}

template< typename Matrix,
          typename Communicator >
const Matrix&
DistributedMatrix< Matrix, Communicator >::
getLocalMatrix() const
{
   return localMatrix;
}

template< typename Matrix,
          typename Communicator >
Matrix&
DistributedMatrix< Matrix, Communicator >::
getLocalMatrix()
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
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getRows() const
{
   return rows;
}

template< typename Matrix,
          typename Communicator >
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getColumns() const
{
   return localMatrix.getColumns();
}

template< typename Matrix,
          typename Communicator >
   template< typename RowCapacitiesVector >
void
DistributedMatrix< Matrix, Communicator >::
setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), getRows(), "row lengths vector has wrong size" );
   TNL_ASSERT_EQ( rowCapacities.getLocalRange(), getLocalRowRange(), "row lengths vector has wrong distribution" );
   TNL_ASSERT_EQ( rowCapacities.getCommunicationGroup(), getCommunicationGroup(), "row lengths vector has wrong communication group" );

   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      localMatrix.setRowCapacities( rowCapacities.getConstLocalView() );

      spmv.reset();
   }
}

template< typename Matrix,
          typename Communicator >
   template< typename Vector >
void
DistributedMatrix< Matrix, Communicator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      rowLengths.setDistribution( getLocalRowRange(), 0, getRows(), getCommunicationGroup() );
      auto localRowLengths = rowLengths.getLocalView();
      localMatrix.getCompressedRowLengths( localRowLengths );
   }
}

template< typename Matrix,
          typename Communicator >
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator >::
getRowCapacity( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRowCapacity( localRow );
}

template< typename Matrix,
          typename Communicator >
void
DistributedMatrix< Matrix, Communicator >::
setElement( IndexType row,
            IndexType column,
            RealType value )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   localMatrix.setElement( localRow, column, value );
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
typename DistributedMatrix< Matrix, Communicator >::MatrixRow
DistributedMatrix< Matrix, Communicator >::
getRow( IndexType row )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix,
          typename Communicator >
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
   TNL_ASSERT_EQ( inVector.getLocalRange(), getLocalRowRange(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicationGroup(), getCommunicationGroup(), "input vector has wrong communication group" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   if( getCommunicationGroup() == CommunicatorType::NullGroup )
      return;

   if( inVector.getGhosts() == 0 ) {
      // NOTE: this branch is deprecated and kept only due to existing benchmarks
      TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
      const_cast< DistributedMatrix* >( this )->spmv.vectorProduct( outVector, localMatrix, localRowRange, inVector, getCommunicationGroup() );
   }
   else {
      TNL_ASSERT_EQ( inVector.getConstLocalViewWithGhosts().getSize(), localMatrix.getColumns(), "the matrix uses non-local and non-ghost column indices" );
      TNL_ASSERT_EQ( inVector.getGhosts(), localMatrix.getColumns() - localMatrix.getRows(), "input vector has wrong ghosts size" );
      TNL_ASSERT_EQ( outVector.getGhosts(), localMatrix.getColumns() - localMatrix.getRows(), "output vector has wrong ghosts size" );
      TNL_ASSERT_EQ( outVector.getConstLocalView().getSize(), localMatrix.getRows(), "number of local matrix rows does not match the output vector local size" );

      inVector.waitForSynchronization();
      const auto inView = inVector.getConstLocalViewWithGhosts();
      auto outView = outVector.getLocalView();
      localMatrix.vectorProduct( inView, outView );
      // TODO: synchronization is not always necessary, e.g. when a preconditioning step follows
//      outVector.startSynchronization();
   }
}

template< typename Matrix,
          typename Communicator >
   template< typename Vector1, typename Vector2 >
bool
DistributedMatrix< Matrix, Communicator >::
performSORIteration( const Vector1& b,
                     const IndexType row,
                     Vector2& x,
                     const RealType& omega ) const
{
   return getLocalMatrix().performSORIteration( b, row, x, omega );
}

} // namespace Matrices
} // namespace TNL
