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

   resetBuffers();
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

   resetBuffers();
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

   resetBuffers();
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

   if( getCommunicationGroup() != CommunicatorType::NullGroup ) {
      localMatrix.setCompressedRowLengths( rowLengths.getLocalVectorView() );

      resetBuffers();
   }
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

template< typename Matrix,
          typename Communicator,
          typename IndexMap >
   template< typename Partitioner >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
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
          typename Communicator,
          typename IndexMap >
   template< typename Partitioner,
             typename RealIn,
             typename RealOut >
void
DistributedMatrix< Matrix, Communicator, IndexMap >::
vectorProduct( const DistVector< RealIn >& inVector,
               DistVector< RealOut >& outVector )
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( inVector.getIndexMap(), getRowIndexMap(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicationGroup(), getCommunicationGroup(), "input vector has wrong communication group" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getIndexMap(), getRowIndexMap(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicationGroup(), getCommunicationGroup(), "output vector has wrong communication group" );

   if( getCommunicationGroup() == CommunicatorType::NullGroup )
      return;

   const int rank = CommunicatorType::GetRank( getCommunicationGroup() );
   const int nproc = CommunicatorType::GetSize( getCommunicationGroup() );

   // update communication pattern
   if( commPattern.getRows() != nproc )
      updateVectorProductCommunicationPattern< Partitioner >();

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
