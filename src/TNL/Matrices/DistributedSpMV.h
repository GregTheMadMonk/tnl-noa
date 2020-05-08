/***************************************************************************
                          DistributedSpMV.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedVectorView.h>

// buffers
#include <vector>
#include <utility>  // std::pair
#include <limits>   // std::numeric_limits
#include <TNL/Allocators/Host.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/ThreePartVector.h>

// operations
#include <type_traits>  // std::add_const
#include <TNL/Atomic.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>

namespace TNL {
namespace Matrices {

template< typename Matrix, typename Communicator >
class DistributedSpMV
{
public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using CommunicatorType = Communicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using Partitioner = Containers::Partitioner< typename Matrix::IndexType, Communicator >;

   // - communication pattern: vector components whose indices are in the range
   //   [start_ij, end_ij) are copied from the j-th process to the i-th process
   //   (an empty range with start_ij == end_ij indicates that there is no
   //   communication between the i-th and j-th processes)
   // - communication pattern matrices - we need to assemble two nproc x nproc
   //   matrices commPatternStarts and commPatternEnds holding the values
   //   start_ij and end_ij respectively
   // - assembly of the i-th row involves traversal of the local matrix stored
   //   in the i-th process
   // - assembly of the full matrix needs all-to-all communication
   void updateCommunicationPattern( const MatrixType& localMatrix, CommunicationGroup group )
   {
      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );
      commPatternStarts.setDimensions( nproc, nproc );
      commPatternEnds.setDimensions( nproc, nproc );

      // pass the localMatrix to the device
      const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );

      // buffer for the local row of the commPattern matrix
      using AtomicIndex = Atomic< IndexType, DeviceType >;
      Containers::Array< AtomicIndex, DeviceType > span_starts( nproc ), span_ends( nproc );
      span_starts.setValue( std::numeric_limits<IndexType>::max() );
      span_ends.setValue( 0 );

      // optimization for banded matrices
      using AtomicIndex = Atomic< IndexType, DeviceType >;
      Containers::Array< AtomicIndex, DeviceType > local_span( 2 );
      local_span.setElement( 0, 0 );  // span start
      local_span.setElement( 1, localMatrix.getRows() );  // span end

      auto kernel = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix,
                                            AtomicIndex* span_starts, AtomicIndex* span_ends, AtomicIndex* local_span )
      {
         const IndexType columns = localMatrix->getColumns();
         const auto row = localMatrix->getRow( i );
         bool comm_left = false;
         bool comm_right = false;
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            const IndexType j = row.getColumnIndex( c );
            if( j < columns ) {
               const int owner = Partitioner::getOwner( j, columns, nproc );
               // atomic assignment
               span_starts[ owner ].fetch_min( j );
               span_ends[ owner ].fetch_max( j + 1 );
               // update comm_left/right
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

      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localMatrix.getRows(),
                                                   kernel,
                                                   &localMatrixPointer.template getData< DeviceType >(),
                                                   span_starts.getData(),
                                                   span_ends.getData(),
                                                   local_span.getData()
                                                );

      // set the local-only span (optimization for banded matrices)
      localOnlySpan.first = local_span.getElement( 0 );
      localOnlySpan.second = local_span.getElement( 1 );

      // copy the buffer into all rows of the preCommPattern* matrices
      // (in-place copy does not work with some OpenMPI configurations)
      Matrices::DenseMatrix< IndexType, Devices::Host, int > preCommPatternStarts, preCommPatternEnds;
      preCommPatternStarts.setLike( commPatternStarts );
      preCommPatternEnds.setLike( commPatternEnds );
      for( int j = 0; j < nproc; j++ )
      for( int i = 0; i < nproc; i++ ) {
         preCommPatternStarts.setElement( j, i, span_starts.getElement( i ) );
         preCommPatternEnds.setElement( j, i, span_ends.getElement( i ) );
      }

      // assemble the commPattern* matrices
      CommunicatorType::Alltoall( &preCommPatternStarts(0, 0), nproc,
                                  &commPatternStarts(0, 0), nproc,
                                  group );
      CommunicatorType::Alltoall( &preCommPatternEnds(0, 0), nproc,
                                  &commPatternEnds(0, 0), nproc,
                                  group );
   }

   template< typename InVector,
             typename OutVector >
   void vectorProduct( OutVector& outVector,
                       const MatrixType& localMatrix,
                       const InVector& inVector,
                       CommunicationGroup group )
   {
      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );

      // update communication pattern
      if( commPatternStarts.getRows() != nproc || commPatternEnds.getRows() != nproc )
         updateCommunicationPattern( localMatrix, group );

      // prepare buffers
      globalBuffer.init( Partitioner::getOffset( localMatrix.getColumns(), rank, nproc ),
                         inVector.getConstLocalView(),
                         localMatrix.getColumns() - Partitioner::getOffset( localMatrix.getColumns(), rank, nproc ) - inVector.getConstLocalView().getSize() );
      const auto globalBufferView = globalBuffer.getConstView();

      // buffer for asynchronous communication requests
      std::vector< typename CommunicatorType::Request > commRequests;

      // send our data to all processes that need it
      for( int i = 0; i < commPatternStarts.getRows(); i++ ) {
         if( i == rank )
             continue;
         if( commPatternStarts( i, rank ) < commPatternEnds( i, rank ) )
            commRequests.push_back( CommunicatorType::ISend(
                     inVector.getConstLocalView().getData() + commPatternStarts( i, rank ) - Partitioner::getOffset( localMatrix.getColumns(), rank, nproc ),
                     commPatternEnds( i, rank ) - commPatternStarts( i, rank ),
                     i, 0, group ) );
      }

      // receive data that we need
      for( int j = 0; j < commPatternStarts.getRows(); j++ ) {
         if( j == rank )
             continue;
         if( commPatternStarts( rank, j ) < commPatternEnds( rank, j ) )
            commRequests.push_back( CommunicatorType::IRecv(
                     globalBuffer.getPointer( commPatternStarts( rank, j ) ),
                     commPatternEnds( rank, j ) - commPatternStarts( rank, j ),
                     j, 0, group ) );
      }

      // general variant
      if( localOnlySpan.first >= localOnlySpan.second ) {
         // wait for all communications to finish
         CommunicatorType::WaitAll( &commRequests[0], commRequests.size() );

         // perform matrix-vector multiplication
         localMatrix.vectorProduct( globalBuffer, outVector );
         /*auto outVectorView = outVector.getLocalView();
         const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );
         auto kernel = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix ) mutable
         {
            outVectorView[ i ] = localMatrix->rowVectorProduct( i, globalBufferView );
         };
         Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localMatrix.getRows(), kernel,
                                                      &localMatrixPointer.template getData< DeviceType >() );*/
      }
      // optimization for banded matrices
      else {
         return;
         auto outVectorView = outVector.getLocalView();
         const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );
         //const auto inView = inVector.getConstView();

         // matrix-vector multiplication using local-only rows
         localMatrix.vectorProduct( inVector, outVector, 1.0, 0.0, localOnlySpan.first, localOnlySpan.second );
         /*auto kernel1 = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix ) mutable
         {
            outVectorView[ i ] = localMatrix->rowVectorProduct( i, inView );
         };
         Algorithms::ParallelFor< DeviceType >::exec( localOnlySpan.first, localOnlySpan.second, kernel1,
                                                      &localMatrixPointer.template getData< DeviceType >() );*/


         // wait for all communications to finish
         CommunicatorType::WaitAll( &commRequests[0], commRequests.size() );

         // finish the multiplication by adding the non-local entries
         localMatrix.vectorProduct( globalBufferView, outVector, 1.0, 0.0, 0, localOnlySpan.first );
         localMatrix.vectorProduct( globalBufferView, outVector, 1.0, 0.0, localOnlySpan.second, localMatrix.getRows() );
         /*auto kernel2 = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix ) mutable
         {
            outVectorView[ i ] = localMatrix->rowVectorProduct( i, globalBufferView );
         };
         Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localOnlySpan.first, kernel2,
                                                      &localMatrixPointer.template getData< DeviceType >() );
         Algorithms::ParallelFor< DeviceType >::exec( localOnlySpan.second, localMatrix.getRows(), kernel2,
                                                      &localMatrixPointer.template getData< DeviceType >() );*/
      }
   }

   void reset()
   {
      commPatternStarts.reset();
      commPatternEnds.reset();
      localOnlySpan.first = localOnlySpan.second = 0;
      globalBuffer.reset();
   }

protected:
   // communication pattern
   Matrices::DenseMatrix< IndexType, Devices::Host, int, true, Allocators::Host< IndexType > > commPatternStarts, commPatternEnds;

   // span of rows with only block-diagonal entries
   std::pair< IndexType, IndexType > localOnlySpan;

   // global buffer for non-local elements of the vector
   __DistributedSpMV_impl::ThreePartVector< RealType, DeviceType, IndexType > globalBuffer;
};

} // namespace Matrices
} // namespace TNL
