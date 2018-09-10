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
namespace DistributedContainers {

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
DistributedMatrix< Matrix, Communicator, IndexMap >::
DistributedMatrix( IndexMap rowIndexMap, IndexType columns, CommunicationGroup group )
{
   setDistribution( rowIndexMap, columns, group );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
setDistribution( IndexMap rowIndexMap, IndexType columns, CommunicationGroup group )
{
   this->rowIndexMap = rowIndexMap;
   this->group = group;
   if( group != Communicator::NullGroup )
      localMatrix.setDimensions( rowIndexMap.getLocalSize(), columns );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
const IndexMap&
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRowIndexMap() const
{
   return rowIndexMap;
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
typename Communicator::CommunicationGroup
DistributedMatrix< Matrix, Communicator, IndexMap >::
getCommunicationGroup() const
{
   return group;
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
const Matrix&
DistributedMatrix< Matrix, Communicator, IndexMap >::
getLocalMatrix() const
{
   return localMatrix;
}


template< typename Matrix,
          typename Communicator,
          typename IndexMap >
String
DistributedMatrix< Matrix, Communicator, IndexMap >::
getType()
{
   return String( "DistributedContainers::DistributedMatrix< " ) +
          Matrix::getType() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator>, " +
          IndexMap::getType() + " >";
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
String
DistributedMatrix< Matrix, Communicator, IndexMap >::
getTypeVirtual() const
{
   return getType();
}


/*
 * Some common Matrix methods follow below.
 */

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
DistributedMatrix< Matrix, Communicator, IndexMap >&
DistributedMatrix< Matrix, Communicator, IndexMap >::
operator=( const DistributedMatrix& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
   template< typename MatrixT >
DistributedMatrix< Matrix, Communicator, IndexMap >&
DistributedMatrix< Matrix, Communicator, IndexMap >::
operator=( const MatrixT& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
   template< typename MatrixT >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
setLike( const MatrixT& matrix )
{
   rowIndexMap = matrix.getRowIndexMap();
   group = matrix.getCommunicationGroup();
   localMatrix.setLike( matrix.getLocalMatrix() );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
reset()
{
   rowIndexMap.reset();
   group = Communicator::NullGroup;
   localMatrix.reset();
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRows() const
{
   return rowIndexMap.getGlobalSize();
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator, IndexMap >::
getColumns() const
{
   return localMatrix.getColumns();
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths )
{
   TNL_ASSERT_EQ( rowLengths.getSize(), getRows(), "row lengths vector has wrong size" );
   TNL_ASSERT_EQ( rowLengths.getIndexMap(), getRowIndexMap(), "row lengths vector has wrong distribution" );
   TNL_ASSERT_EQ( rowLengths.getCommunicationGroup(), getCommunicationGroup(), "row lengths vector has wrong communication group" );

   if( getCommunicationGroup() != CommunicatorType::NullGroup )
      localMatrix.setCompressedRowLengths( rowLengths.getLocalVectorView() );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const
{
   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      rowLengths.setDistribution( getRowIndexMap(), getCommunicationGroup() );
      localMatrix.getCompressedRowLengths( rowLengths.getLocalVectorView() );
   }
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
typename Matrix::IndexType
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRowLength( IndexType row ) const
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getRowLength( localRow );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
bool
DistributedMatrix< Matrix, Communicator, IndexMap >::
setElement( IndexType row,
            IndexType column,
            RealType value )
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.setElement( localRow, column, value );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
bool
DistributedMatrix< Matrix, Communicator, IndexMap >::
setElementFast( IndexType row,
                IndexType column,
                RealType value )
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.setElementFast( localRow, column, value );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
typename Matrix::RealType
DistributedMatrix< Matrix, Communicator, IndexMap >::
getElement( IndexType row,
            IndexType column ) const
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getElement( localRow, column );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
typename Matrix::RealType
DistributedMatrix< Matrix, Communicator, IndexMap >::
getElementFast( IndexType row,
                IndexType column ) const
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getElementFast( localRow, column );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
bool
DistributedMatrix< Matrix, Communicator, IndexMap >::
setRowFast( IndexType row,
            const IndexType* columnIndexes,
            const RealType* values,
            IndexType elements )
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.setRowFast( localRow, columnIndexes, values, elements );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRowFast( IndexType row,
            IndexType* columns,
            RealType* values ) const
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getRowFast( localRow, columns, values );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
typename DistributedMatrix< Matrix, Communicator, IndexMap >::MatrixRow
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRow( IndexType row )
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
__cuda_callable__
typename DistributedMatrix< Matrix, Communicator, IndexMap >::ConstMatrixRow
DistributedMatrix< Matrix, Communicator, IndexMap >::
getRow( IndexType row ) const
{
   const IndexType localRow = rowIndexMap.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
   template< typename Vector,
             typename RealOut >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
vectorProduct( const Vector& inVector,
               DistVector< RealOut >& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getIndexMap(), getRowIndexMap(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   auto outView = outVector.getLocalVectorView();
   localMatrix.vectorProduct( inVector, outView );
}

} // namespace DistributedContainers
} // namespace TNL
