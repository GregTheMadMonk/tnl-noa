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

#include <TNL/DistributedContainers/Partitioner.h>
#include <TNL/DistributedContainers/DistributedVectorView.h>

// buffers
#include <vector>
#include <utility>  // std::pair
#include <TNL/Matrices/Dense.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

// operations
#include <type_traits>  // std::add_const
#include <TNL/Atomic.h>
#include <TNL/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>

namespace TNL {
namespace DistributedContainers {

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
   using Partitioner = DistributedContainers::Partitioner< typename Matrix::IndexType, Communicator >;

   // - communication pattern matrix is an nproc x nproc binary matrix C, where
   //   C_ij = 1 iff the i-th process needs data from the j-th process
   // - assembly of the i-th row involves traversal of the local matrix stored
   //   in the i-th process
   // - assembly the full matrix needs all-to-all communication
   void updateCommunicationPattern( const MatrixType& localMatrix, CommunicationGroup group )
   {
      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );
      commPattern.setDimensions( nproc, nproc );

      // pass the localMatrix to the device
      const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );

      // buffer for the local row of the commPattern matrix
//      using AtomicBool = Atomic< bool, DeviceType >;
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
      if( commPattern.getRows() != nproc )
         updateCommunicationPattern( localMatrix, group );

      // prepare buffers
      globalBuffer.setSize( localMatrix.getColumns() );
      commRequests.clear();

      // send our data to all processes that need it
      for( int i = 0; i < commPattern.getRows(); i++ )
         if( commPattern( i, rank ) )
            commRequests.push_back( CommunicatorType::ISend(
                     inVector.getLocalVectorView().getData(),
                     inVector.getLocalVectorView().getSize(),
                     i, group ) );

      // receive data that we need
      for( int j = 0; j < commPattern.getRows(); j++ )
         if( commPattern( rank, j ) )
            commRequests.push_back( CommunicatorType::IRecv(
                     &globalBuffer[ Partitioner::getOffset( globalBuffer.getSize(), j, nproc ) ],
                     Partitioner::getSizeForRank( globalBuffer.getSize(), j, nproc ),
                     j, group ) );

      // general variant
      if( localOnlySpan.first >= localOnlySpan.second ) {
         // wait for all communications to finish
         CommunicatorType::WaitAll( &commRequests[0], commRequests.size() );

         // perform matrix-vector multiplication
         auto outView = outVector.getLocalVectorView();
         localMatrix.vectorProduct( globalBuffer, outView );
      }
      // optimization for banded matrices
      else {
         auto outVectorView = outVector.getLocalVectorView();
         const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );
         using InView = DistributedVectorView< const typename InVector::RealType, typename InVector::DeviceType, typename InVector::IndexType, typename InVector::CommunicatorType >;
         const InView inView( inVector );

         // matrix-vector multiplication using local-only rows
         auto kernel1 = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix ) mutable
         {
            outVectorView[ i ] = localMatrix->rowVectorProduct( i, inView );
         };
         ParallelFor< DeviceType >::exec( localOnlySpan.first, localOnlySpan.second, kernel1,
                                          &localMatrixPointer.template getData< DeviceType >() );

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

   void reset()
   {
      commPattern.reset();
      localOnlySpan.first = localOnlySpan.second = 0;
      globalBuffer.reset();
      commRequests.clear();
   }

protected:
   // communication pattern
   Matrices::Dense< bool, Devices::Host, int > commPattern;

   // span of rows with only block-diagonal entries
   std::pair< IndexType, IndexType > localOnlySpan;

   // global buffer for non-local elements of the vector
   Containers::Vector< RealType, DeviceType, IndexType > globalBuffer;

   // buffer for asynchronous communication requests
   std::vector< typename CommunicatorType::Request > commRequests;
};

} // namespace DistributedContainers
} // namespace TNL
