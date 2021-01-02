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

template< typename Matrix >
DistributedMatrix< Matrix >::
DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm group )
{
   setDistribution( localRowRange, rows, columns, group );
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm group )
{
   this->localRowRange = localRowRange;
   this->rows = rows;
   this->group = group;
   if( group != MPI::NullGroup() )
      localMatrix.setDimensions( localRowRange.getSize(), columns );

   spmv.reset();
}

template< typename Matrix >
const Containers::Subrange< typename Matrix::IndexType >&
DistributedMatrix< Matrix >::
getLocalRowRange() const
{
   return localRowRange;
}

template< typename Matrix >
MPI_Comm
DistributedMatrix< Matrix >::
getCommunicationGroup() const
{
   return group;
}

template< typename Matrix >
const Matrix&
DistributedMatrix< Matrix >::
getLocalMatrix() const
{
   return localMatrix;
}

template< typename Matrix >
Matrix&
DistributedMatrix< Matrix >::
getLocalMatrix()
{
   return localMatrix;
}


/*
 * Some common Matrix methods follow below.
 */

template< typename Matrix >
DistributedMatrix< Matrix >&
DistributedMatrix< Matrix >::
operator=( const DistributedMatrix& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix >
   template< typename MatrixT >
DistributedMatrix< Matrix >&
DistributedMatrix< Matrix >::
operator=( const MatrixT& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix >
   template< typename MatrixT >
void
DistributedMatrix< Matrix >::
setLike( const MatrixT& matrix )
{
   localRowRange = matrix.getLocalRowRange();
   rows = matrix.getRows();
   group = matrix.getCommunicationGroup();
   localMatrix.setLike( matrix.getLocalMatrix() );

   spmv.reset();
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
reset()
{
   localRowRange.reset();
   rows = 0;
   group = MPI::NullGroup();
   localMatrix.reset();

   spmv.reset();
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getRows() const
{
   return rows;
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getColumns() const
{
   return localMatrix.getColumns();
}

template< typename Matrix >
   template< typename RowCapacitiesVector >
void
DistributedMatrix< Matrix >::
setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), getRows(), "row lengths vector has wrong size" );
   TNL_ASSERT_EQ( rowCapacities.getLocalRange(), getLocalRowRange(), "row lengths vector has wrong distribution" );
   TNL_ASSERT_EQ( rowCapacities.getCommunicationGroup(), getCommunicationGroup(), "row lengths vector has wrong communication group" );

   if( getCommunicationGroup() != MPI::NullGroup() ) {
      localMatrix.setRowCapacities( rowCapacities.getConstLocalView() );

      spmv.reset();
   }
}

template< typename Matrix >
   template< typename Vector >
void
DistributedMatrix< Matrix >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   if( getCommunicationGroup() != MPI::NullGroup() ) {
      rowLengths.setDistribution( getLocalRowRange(), 0, getRows(), getCommunicationGroup() );
      auto localRowLengths = rowLengths.getLocalView();
      localMatrix.getCompressedRowLengths( localRowLengths );
   }
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getRowCapacity( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRowCapacity( localRow );
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
setElement( IndexType row,
            IndexType column,
            RealType value )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   localMatrix.setElement( localRow, column, value );
}

template< typename Matrix >
typename Matrix::RealType
DistributedMatrix< Matrix >::
getElement( IndexType row,
            IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElement( localRow, column );
}

template< typename Matrix >
typename Matrix::RealType
DistributedMatrix< Matrix >::
getElementFast( IndexType row,
                IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElementFast( localRow, column );
}

template< typename Matrix >
typename DistributedMatrix< Matrix >::MatrixRow
DistributedMatrix< Matrix >::
getRow( IndexType row )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix >
typename DistributedMatrix< Matrix >::ConstMatrixRow
DistributedMatrix< Matrix >::
getRow( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< ! HasGetCommunicationGroupMethod< InVector >::value >::type
DistributedMatrix< Matrix >::
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

template< typename Matrix >
void
DistributedMatrix< Matrix >::
updateVectorProductCommunicationPattern()
{
   if( getCommunicationGroup() == MPI::NullGroup() )
      return;
   spmv.updateCommunicationPattern( getLocalMatrix(), getCommunicationGroup() );
}

template< typename Matrix >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< HasGetCommunicationGroupMethod< InVector >::value >::type
DistributedMatrix< Matrix >::
vectorProduct( const InVector& inVector,
               OutVector& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getLocalRange(), getLocalRowRange(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicationGroup(), getCommunicationGroup(), "input vector has wrong communication group" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   if( getCommunicationGroup() == MPI::NullGroup() )
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

template< typename Matrix >
   template< typename Vector1, typename Vector2 >
bool
DistributedMatrix< Matrix >::
performSORIteration( const Vector1& b,
                     const IndexType row,
                     Vector2& x,
                     const RealType& omega ) const
{
   return getLocalMatrix().performSORIteration( b, row, x, omega );
}

} // namespace Matrices
} // namespace TNL
