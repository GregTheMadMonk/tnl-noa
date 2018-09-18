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

#include <TNL/Atomic.h>
#include <TNL/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace DistributedContainers {

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

   resetBuffers();
}

template< typename Matrix,
          typename Communicator >
const Subrange< typename Matrix::IndexType >&
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
String
DistributedMatrix< Matrix, Communicator >::
getType()
{
   return String( "DistributedContainers::DistributedMatrix< " ) +
          Matrix::getType() + ", " +
          // TODO: communicators don't have a getType method
          "<Communicator>" + " >";
}

template< typename Matrix,
          typename Communicator >
String
DistributedMatrix< Matrix, Communicator >::
getTypeVirtual() const
{
   return getType();
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

   resetBuffers();
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

   resetBuffers();
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
      localMatrix.setCompressedRowLengths( rowLengths.getLocalVectorView() );

      resetBuffers();
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
      localMatrix.getCompressedRowLengths( rowLengths.getLocalVectorView() );
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
   template< typename Vector,
             typename RealOut >
void
DistributedMatrix< Matrix, Communicator >::
vectorProduct( const Vector& inVector,
               DistVector< RealOut >& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   auto outView = outVector.getLocalVectorView();
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

   const int rank = CommunicatorType::GetRank( getCommunicationGroup() );
   const int nproc = CommunicatorType::GetSize( getCommunicationGroup() );
   commPattern.setDimensions( nproc, nproc );

   // pass the localMatrix to the device
   Pointers::DevicePointer< MatrixType > localMatrixPointer( localMatrix );

   // buffer for the local row of the commPattern matrix
//   using AtomicBool = Atomic< bool, DeviceType >;
   // FIXME: CUDA does not support atomic operations for bool
   using AtomicBool = Atomic< int, DeviceType >;
   Containers::Array< AtomicBool, DeviceType > buffer( nproc );
   buffer.setValue( false );

   // optimization for banded matrices
   using AtomicIndex = Atomic< IndexType, DeviceType >;
   Containers::Array< AtomicIndex, DeviceType > local_span( 2 );
   local_span.setElement( 0, 0 );  // span start
   local_span.setElement( 1, localMatrix.getRows() );  // span end

   auto kernel = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix,
                                         AtomicBool* buffer, AtomicIndex* local_span )
   {
      const IndexType columns = localMatrix->getColumns();
      const auto row = localMatrix->getRow( i );
      bool comm_left = false;
      bool comm_right = false;
      for( IndexType c = 0; c < row.getLength(); c++ ) {
         const IndexType j = row.getElementColumn( c );
         if( j < columns ) {
            const int owner = Partitioner::getOwner( j, columns, nproc );
            // atomic assignment
            buffer[ owner ].store( true );
            // update comm_left/Right
            if( owner < rank )
               comm_left = true;
            if( owner > rank )
               comm_right = true;
         }
      }
      // update local span
      if( comm_left )
         local_span[0].fetch_max( i + 1 );
      if( comm_right )
         local_span[1].fetch_min( i );
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, localMatrix.getRows(),
                                    kernel,
                                    &localMatrixPointer.template getData< DeviceType >(),
                                    buffer.getData(),
                                    local_span.getData()
                                 );

   // set the local-only span (optimization for banded matrices)
   localOnlySpan.first = local_span.getElement( 0 );
   localOnlySpan.second = local_span.getElement( 1 );

   // copy the buffer into all rows of the preCommPattern matrix
   Matrices::Dense< bool, Devices::Host, int > preCommPattern;
   preCommPattern.setLike( commPattern );
   for( int j = 0; j < nproc; j++ )
   for( int i = 0; i < nproc; i++ )
      preCommPattern.setElementFast( j, i, buffer.getElement( i ) );

   // assemble the commPattern matrix
   CommunicatorType::Alltoall( &preCommPattern(0, 0), nproc,
                               &commPattern(0, 0), nproc,
                               getCommunicationGroup() );
}

template< typename Matrix,
          typename Communicator >
   template< typename RealIn,
             typename RealOut >
void
DistributedMatrix< Matrix, Communicator >::
vectorProduct( const DistVector< RealIn >& inVector,
               DistVector< RealOut >& outVector )
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( inVector.getLocalRange(), getLocalRowRange(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicationGroup(), getCommunicationGroup(), "input vector has wrong communication group" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   if( getCommunicationGroup() == CommunicatorType::NullGroup )
      return;

   const int rank = CommunicatorType::GetRank( getCommunicationGroup() );
   const int nproc = CommunicatorType::GetSize( getCommunicationGroup() );

   // update communication pattern
   if( commPattern.getRows() != nproc )
      updateVectorProductCommunicationPattern();

   // prepare buffers
   globalBuffer.setSize( localMatrix.getColumns() );
   commRequests.clear();

   // send our data to all processes that need it
   for( int i = 0; i < commPattern.getRows(); i++ )
      if( commPattern( i, rank ) )
         commRequests.push_back( CommunicatorType::ISend(
                  inVector.getLocalVectorView().getData(),
                  inVector.getLocalVectorView().getSize(),
                  i, getCommunicationGroup() ) );

   // receive data that we need
   for( int j = 0; j < commPattern.getRows(); j++ )
      if( commPattern( rank, j ) )
         commRequests.push_back( CommunicatorType::IRecv(
                  &globalBuffer[ Partitioner::getOffset( globalBuffer.getSize(), j, nproc ) ],
                  Partitioner::getSizeForRank( globalBuffer.getSize(), j, nproc ),
                  j, getCommunicationGroup() ) );

   // general variant
   if( localOnlySpan.first >= localOnlySpan.second ) {
      // wait for all communications to finish
      CommunicatorType::WaitAll( &commRequests[0], commRequests.size() );

      // perform matrix-vector multiplication
      vectorProduct( globalBuffer, outVector );
   }
   // optimization for banded matrices
   else {
      Pointers::DevicePointer< MatrixType > localMatrixPointer( localMatrix );
      auto outVectorView = outVector.getLocalVectorView();
      // TODO
//      const auto inVectorView = DistributedVectorView( inVector );
      Pointers::DevicePointer< const DistVector< RealIn > > inVectorPointer( inVector );

      // matrix-vector multiplication using local-only rows
      auto kernel1 = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix, const DistVector< RealIn >* inVector ) mutable
      {
         outVectorView[ i ] = localMatrix->rowVectorProduct( i, *inVector );
      };
      ParallelFor< DeviceType >::exec( localOnlySpan.first, localOnlySpan.second, kernel1,
                                       &localMatrixPointer.template getData< DeviceType >(),
                                       &inVectorPointer.template getData< DeviceType >() );

      // wait for all communications to finish
      CommunicatorType::WaitAll( &commRequests[0], commRequests.size() );

      // finish the multiplication by adding the non-local entries
      Containers::VectorView< RealType, DeviceType, IndexType > globalBufferView( globalBuffer );
      auto kernel2 = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix ) mutable
      {
         outVectorView[ i ] = localMatrix->rowVectorProduct( i, globalBufferView );
      };
      ParallelFor< DeviceType >::exec( (IndexType) 0, localOnlySpan.first, kernel2,
                                       &localMatrixPointer.template getData< DeviceType >() );
      ParallelFor< DeviceType >::exec( localOnlySpan.second, localMatrix.getRows(), kernel2,
                                       &localMatrixPointer.template getData< DeviceType >() );
   }
}

} // namespace DistributedContainers
} // namespace TNL
