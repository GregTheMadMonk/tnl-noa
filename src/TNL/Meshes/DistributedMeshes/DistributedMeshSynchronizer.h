/***************************************************************************
                          DistributedMeshSynchronizer.h  -  description
                             -------------------
    begin                : April 12, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename DistributedMesh,
          int EntityDimension = DistributedMesh::getMeshDimension() >
class DistributedMeshSynchronizer
{
public:
   using DeviceType = typename DistributedMesh::DeviceType;
   using GlobalIndexType = typename DistributedMesh::GlobalIndexType;
   using CommunicatorType = typename DistributedMesh::CommunicatorType;

   DistributedMeshSynchronizer() = default;

   void initialize( const DistributedMesh& mesh )
   {
      if( mesh.getGhostLevels() <= 0 )
         throw std::logic_error( "There are no ghost levels on the distributed mesh." );
      TNL_ASSERT_EQ( mesh.template getGlobalIndices< EntityDimension >().getSize(), mesh.getLocalMesh().template getEntitiesCount< EntityDimension >(),
                     "Global indices are not allocated properly." );

      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      #ifdef HAVE_CUDA
      if( std::is_same< DeviceType, Devices::Cuda >::value )
         cudaGetDevice(&this->gpu_id);
      #endif

      group = mesh.getCommunicationGroup();
      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );

      // exchange the global index offsets so that each rank can determine the
      // owner of every entity by its global index
      const GlobalIndexType ownStart = mesh.template getGlobalIndices< EntityDimension >().getElement( 0 );
      Containers::Array< GlobalIndexType, Devices::Host, int > offsets( nproc );
      {
         Containers::Array< GlobalIndexType, Devices::Host, int > sendbuf( nproc );
         sendbuf.setValue( ownStart );
         CommunicatorType::Alltoall( sendbuf.getData(), 1,
                                     offsets.getData(), 1,
                                     group );
      }

      auto getOwner = [&] ( GlobalIndexType idx )
      {
         for( int i = 0; i < nproc - 1; i++ )
            if( offsets[ i ] <= idx && idx < offsets[ i + 1 ] )
               return i;
         return nproc - 1;
      };

      // count local ghost entities for each rank
      Containers::Array< GlobalIndexType, Devices::Host, int > localGhostCounts( nproc );
      {
         localGhostCounts.setValue( 0 );
         Containers::Array< GlobalIndexType, Devices::Host, GlobalIndexType > hostGlobalIndices;
         hostGlobalIndices = mesh.template getGlobalIndices< EntityDimension >();
         GlobalIndexType prev_global_idx = 0;
         mesh.getLocalMesh().template forGhost< EntityDimension, Devices::Sequential >(
            [&] ( GlobalIndexType local_idx )
            {
               if( ! std::is_same< DeviceType, Devices::Cuda >::value )
               if( ! mesh.getLocalMesh().template isGhostEntity< EntityDimension >( local_idx ) )
                  throw std::runtime_error( "encountered local entity while iterating over ghost entities - the mesh is probably inconsistent or there is a bug in the DistributedMeshSynchronizer" );
               const GlobalIndexType global_idx = hostGlobalIndices[ local_idx ];
               if( global_idx < prev_global_idx )
                  throw std::runtime_error( "ghost indices are not sorted - the mesh is probably inconsistent or there is a bug in the DistributedMeshSynchronizer" );
               prev_global_idx = global_idx;
               const int owner = getOwner( global_idx );
               if( owner == rank )
                  throw std::runtime_error( "the owner of a ghost entity cannot be the local rank - the mesh is probably inconsistent or there is a bug in the DistributedMeshSynchronizer" );
               ++localGhostCounts[ owner ];
            }
         );
      }

      // exchange the ghost counts
      ghostEntitiesCounts.setDimensions( nproc, nproc );
      {
         Matrices::DenseMatrix< GlobalIndexType, Devices::Host, int > sendbuf;
         sendbuf.setDimensions( nproc, nproc );
         // copy the local ghost counts into all rows of the sendbuf
         for( int j = 0; j < nproc; j++ )
         for( int i = 0; i < nproc; i++ )
            sendbuf.setElement( j, i, localGhostCounts[ i ] );
         CommunicatorType::Alltoall( &sendbuf(0, 0), nproc,
                                     &ghostEntitiesCounts(0, 0), nproc,
                                     group );
      }

      // allocate ghost offsets
      ghostOffsets.setSize( nproc + 1 );

      // set ghost neighbor offsets
      ghostNeighborOffsets.setSize( nproc + 1 );
      ghostNeighborOffsets[ 0 ] = 0;
      for( int i = 0; i < nproc; i++ )
         ghostNeighborOffsets[ i + 1 ] = ghostNeighborOffsets[ i ] + ghostEntitiesCounts( i, rank );

      // allocate ghost neighbors
      ghostNeighbors.setSize( ghostNeighborOffsets[ nproc ] );

      // send indices of ghost entities - set them as ghost neighbors on the target rank
      {
         std::vector< typename CommunicatorType::Request > requests;

         // send our ghost indices to the neighboring ranks
         GlobalIndexType ghostOffset = mesh.getLocalMesh().template getGhostEntitiesOffset< EntityDimension >();
         ghostOffsets[ 0 ] = ghostOffset;
         for( int i = 0; i < nproc; i++ ) {
            if( ghostEntitiesCounts( rank, i ) > 0 ) {
               requests.push_back( CommunicatorType::ISend(
                        mesh.template getGlobalIndices< EntityDimension >().getData() + ghostOffset,
                        ghostEntitiesCounts( rank, i ),
                        i, 0, group ) );
               ghostOffset += ghostEntitiesCounts( rank, i );
            }
            // update ghost offsets
            ghostOffsets[ i + 1 ] = ghostOffset;
         }
         TNL_ASSERT_EQ( ghostOffsets[ nproc ], mesh.getLocalMesh().template getEntitiesCount< EntityDimension >(),
                        "bug in setting ghost offsets" );

         // receive ghost indices from the neighboring ranks
         for( int j = 0; j < nproc; j++ ) {
            if( ghostEntitiesCounts( j, rank ) > 0 ) {
               requests.push_back( CommunicatorType::IRecv(
                        ghostNeighbors.getData() + ghostNeighborOffsets[ j ],
                        ghostEntitiesCounts( j, rank ),
                        j, 0, group ) );
            }
         }

         // wait for all communications to finish
         CommunicatorType::WaitAll( requests.data(), requests.size() );

         // convert received ghost indices from global to local
         ghostNeighbors -= ownStart;
      }
   }

   template< typename MeshFunction >
   void synchronize( MeshFunction& function )
   {
      static_assert( MeshFunction::getEntitiesDimension() == EntityDimension,
                     "the mesh function's entity dimension does not match" );
      static_assert( std::is_same< typename MeshFunction::MeshType, typename DistributedMesh::MeshType >::value,
                     "The type of the mesh function's mesh does not match the local mesh." );
      TNL_ASSERT_EQ( function.getData().getSize(), ghostOffsets[ ghostOffsets.getSize() - 1 ],
                     "The mesh function does not have the expected size." );
      using RealType = typename MeshFunction::RealType;

      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      #ifdef HAVE_CUDA
      if( std::is_same< DeviceType, Devices::Cuda >::value )
         cudaSetDevice(gpu_id);
      #endif

      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );

      // allocate send buffers (setSize does nothing if the array size is already correct)
      sendBuffers.setSize( ghostNeighborOffsets[ nproc ] * sizeof(RealType) );

      // buffer for asynchronous communication requests
      std::vector< typename CommunicatorType::Request > requests;

      // issue all receive async operations
      for( int j = 0; j < nproc; j++ ) {
         if( ghostEntitiesCounts( rank, j ) > 0 ) {
            requests.push_back( CommunicatorType::IRecv(
                     function.getData().getData() + ghostOffsets[ j ],
                     ghostEntitiesCounts( rank, j ),
                     j, 0, group ) );
         }
      }

      Containers::ArrayView< RealType, DeviceType, GlobalIndexType > sendBuffersView;
      sendBuffersView.bind( reinterpret_cast<RealType*>( sendBuffers.getData() ), ghostNeighborOffsets[ nproc ] );
      const auto ghostNeighborsView = ghostNeighbors.getConstView();
      const auto functionDataView = function.getData().getConstView();
      auto copy_kernel = [sendBuffersView, functionDataView, ghostNeighborsView] __cuda_callable__ ( GlobalIndexType k, GlobalIndexType offset ) mutable
      {
         sendBuffersView[ offset + k ] = functionDataView[ ghostNeighborsView[ offset + k ] ];
      };

      for( int i = 0; i < nproc; i++ ) {
         if( ghostEntitiesCounts( i, rank ) > 0 ) {
            const GlobalIndexType offset = ghostNeighborOffsets[ i ];
            // copy data to send buffers
            Algorithms::ParallelFor< DeviceType >::exec( (GlobalIndexType) 0, ghostEntitiesCounts( i, rank ), copy_kernel, offset );

            // issue async send operation
            requests.push_back( CommunicatorType::ISend(
                     sendBuffersView.getData() + ghostNeighborOffsets[ i ],
                     ghostEntitiesCounts( i, rank ),
                     i, 0, group ) );
         }
      }

      // wait for all communications to finish
      CommunicatorType::WaitAll( requests.data(), requests.size() );
   }

   // performs a synchronization of a sparse matrix
   //    - row indices -- ghost indices (recv), ghost neighbors (send)
   //    - column indices -- values to be synchronized (in a CSR-like format)
   // NOTE: if all shared rows have the same capacity on all ranks that send/receive data
   //       from these rows (e.g. if all row capacities in the matrix are constant),
   //       set assumeConsistentRowCapacities to true
   // returns: a tuple of the received rankOffsets, rowPointers and columnIndices arrays
   template< typename SparsePattern >
   auto
   synchronizeSparse( const SparsePattern& pattern, bool assumeConsistentRowCapacities = false )
   {
      TNL_ASSERT_EQ( pattern.getRows(), ghostOffsets[ ghostOffsets.getSize() - 1 ], "invalid sparse pattern matrix" );

      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );

      // buffer for asynchronous communication requests
      std::vector< typename CommunicatorType::Request > requests;

      Containers::Array< GlobalIndexType, Devices::Host, int > send_rankOffsets( nproc + 1 ), recv_rankOffsets( nproc + 1 );
      Containers::Array< GlobalIndexType, Devices::Host, GlobalIndexType > send_rowCapacities, send_rowPointers, send_columnIndices, recv_rowPointers, recv_columnIndices;

      // sending part
      {
         // set rank offsets
         send_rankOffsets[ 0 ] = 0;
         for( int i = 0; i < nproc; i++ )
            send_rankOffsets[ i + 1 ] = send_rankOffsets[ i ] + ghostNeighborOffsets[ i + 1 ] - ghostNeighborOffsets[ i ];

         // allocate row pointers
         send_rowPointers.setSize( send_rankOffsets[ nproc ] + 1 );
         if( ! assumeConsistentRowCapacities )
            send_rowCapacities.setSize( send_rankOffsets[ nproc ] + 1 );

         // set row pointers
         GlobalIndexType rowPtr = 0;
         send_rowPointers[ rowPtr ] = 0;
         for( int i = 0; i < nproc; i++ ) {
            if( i == rank )
               continue;
            for( GlobalIndexType r = ghostNeighborOffsets[ i ]; r < ghostNeighborOffsets[ i + 1 ]; r++ ) {
               const GlobalIndexType rowIdx = ghostNeighbors[ r ];
               const auto row = pattern.getRow( rowIdx );
               send_rowPointers[ rowPtr + 1 ] = send_rowPointers[ rowPtr ] + row.getSize();
               if( ! assumeConsistentRowCapacities )
                  send_rowCapacities[ rowPtr ] = row.getSize();
               rowPtr++;
            }
            // send our row sizes to the target rank
            if( ! assumeConsistentRowCapacities )
               // issue async send operation
               requests.push_back( CommunicatorType::ISend(
                        send_rowCapacities.getData() + send_rankOffsets[ i ],
                        ghostNeighborOffsets[ i + 1 ] - ghostNeighborOffsets[ i ],
                        i, 1, group ) );
         }

         // allocate column indices
         send_columnIndices.setSize( send_rowPointers[ send_rowPointers.getSize() - 1 ] );

         // set column indices
         for( int i = 0; i < nproc; i++ ) {
            if( i == rank )
               continue;
            const GlobalIndexType rankOffset = send_rankOffsets[ i ];
            GlobalIndexType rowBegin = send_rowPointers[ rankOffset ];
            for( GlobalIndexType r = ghostNeighborOffsets[ i ]; r < ghostNeighborOffsets[ i + 1 ]; r++ ) {
               const GlobalIndexType rowIdx = ghostNeighbors[ r ];
               const auto row = pattern.getRow( rowIdx );
               for( GlobalIndexType c = 0; c < row.getSize(); c++ )
                  send_columnIndices[ rowBegin + c ] = row.getColumnIndex( c );
               rowBegin += row.getSize();
            }
         }

         for( int i = 0; i < nproc; i++ ) {
            if( send_rankOffsets[ i + 1 ] == send_rankOffsets[ i ] )
               continue;
            // issue async send operation
            requests.push_back( CommunicatorType::ISend(
                     send_columnIndices.getData() + send_rowPointers[ send_rankOffsets[ i ] ],
                     send_rowPointers[ send_rankOffsets[ i + 1 ] ] - send_rowPointers[ send_rankOffsets[ i ] ],
                     i, 0, group ) );
         }
      }

      // receiving part
      {
         // set rank offsets
         recv_rankOffsets[ 0 ] = 0;
         for( int i = 0; i < nproc; i++ )
            recv_rankOffsets[ i + 1 ] = recv_rankOffsets[ i ] + ghostOffsets[ i + 1 ] - ghostOffsets[ i ];

         // allocate row pointers
         recv_rowPointers.setSize( recv_rankOffsets[ nproc ] + 1 );

         std::vector< typename CommunicatorType::Request > row_lengths_requests;

         // set row pointers
         GlobalIndexType rowPtr = 0;
         recv_rowPointers[ rowPtr ] = 0;
         for( int i = 0; i < nproc; i++ ) {
            if( i == rank )
               continue;
            if( assumeConsistentRowCapacities ) {
               for( GlobalIndexType rowIdx = ghostOffsets[ i ]; rowIdx < ghostOffsets[ i + 1 ]; rowIdx++ ) {
                  const auto row = pattern.getRow( rowIdx );
                  recv_rowPointers[ rowPtr + 1 ] = recv_rowPointers[ rowPtr ] + row.getSize();
                  rowPtr++;
               }
            }
            else {
               // receive row sizes from the sender
               // issue async recv operation
               row_lengths_requests.push_back( CommunicatorType::IRecv(
                        recv_rowPointers.getData() + recv_rankOffsets[ i ],
                        ghostOffsets[ i + 1 ] - ghostOffsets[ i ],
                        i, 1, group ) );
            }
         }

         if( ! assumeConsistentRowCapacities ) {
            // wait for all row lengths
            CommunicatorType::WaitAll( row_lengths_requests.data(), row_lengths_requests.size() );

            // scan the rowPointers array to convert
            Containers::VectorView< GlobalIndexType, Devices::Host, GlobalIndexType > rowPointersView;
            rowPointersView.bind( recv_rowPointers );
            rowPointersView.template scan< Algorithms::ScanType::Exclusive >();
         }

         // allocate column indices
         recv_columnIndices.setSize( recv_rowPointers[ recv_rowPointers.getSize() - 1 ] );

         for( int i = 0; i < nproc; i++ ) {
            if( recv_rankOffsets[ i + 1 ] == recv_rankOffsets[ i ] )
               continue;
            // issue async recv operation
            requests.push_back( CommunicatorType::IRecv(
                     recv_columnIndices.getData() + recv_rowPointers[ recv_rankOffsets[ i ] ],
                     recv_rowPointers[ recv_rankOffsets[ i + 1 ] ] - recv_rowPointers[ recv_rankOffsets[ i ] ],
                     i, 0, group ) );
         }
      }

      // wait for all communications to finish
      CommunicatorType::WaitAll( requests.data(), requests.size() );

      return std::make_tuple( recv_rankOffsets, recv_rowPointers, recv_columnIndices );
   }

   // public const accessors for the communication pattern matrix and index arrays which were
   // created in the `initialize` method
   const auto& getGhostEntitiesCounts() const
   {
      return ghostEntitiesCounts;
   }

   const auto& getGhostOffsets() const
   {
      return ghostOffsets;
   }

   const auto& getGhostNeighborOffsets() const
   {
      return ghostNeighborOffsets;
   }

   const auto& getGhostNeighbors() const
   {
      return ghostNeighbors;
   }

protected:
   // GOTCHA (see above)
   int gpu_id = 0;

   // communication group taken from the distributed mesh
   typename CommunicatorType::CommunicationGroup group;

   /**
    * Communication pattern:
    * - an unsymmetric nproc x nproc matrix G such that G_ij represents the
    *   number of ghost entities on rank i that are owned by rank j
    * - assembly of the i-th row involves traversal of the ghost entities on the
    *   local mesh and determining its owner based on the global index
    * - assembly of the full matrix needs all-to-all communication
    * - for the i-th rank, the i-th row determines the receive buffer sizes and
    *   the i-th column determines the send buffer sizes
    */
   Matrices::DenseMatrix< GlobalIndexType, Devices::Host, int > ghostEntitiesCounts;

   /**
    * Ghost offsets: array of size nproc + 1 where the i-th value is the local
    * index of the first ghost entity owned by the i-th rank. The last value is
    * equal to the entities count on the local mesh. All ghost entities owned by
    * the i-th rank are assumed to be indexed contiguously in the local mesh.
    */
   Containers::Array< GlobalIndexType, Devices::Host, int > ghostOffsets;

   /**
    * Ghost neighbor offsets: array of size nproc + 1 where the i-th value is
    * the offset of ghost neighbor indices requested by the i-th rank. The last
    * value is the size of the ghostNeighbors and sendBuffers arrays (see
    * below).
    */
   Containers::Array< GlobalIndexType, Devices::Host, int > ghostNeighborOffsets;

   /**
    * Ghost neighbor indices: array containing local indices of the entities
    * which are ghosts on other ranks. The indices requested by the i-th rank
    * are in the range starting at ghostNeighborOffsets[i] (inclusive) and
    * ending at ghostNeighborOffsets[i+1] (exclusive). These indices are used
    * for copying the mesh function values into the sendBuffers array. Note that
    * ghost neighbor indices cannot be made contiguous in general so we need the
    * send buffers.
    */
   Containers::Vector< GlobalIndexType, DeviceType, GlobalIndexType > ghostNeighbors;

   /**
    * Send buffers: array for buffering the mesh function values which will be
    * sent to other ranks. We use std::uint8_t as the value type to make this
    * class independent of the mesh function's real type. When cast to the real
    * type values, the send buffer for the i-th rank is the part of the array
    * starting at index ghostNeighborOffsets[i] (inclusive) and ending at index
    * ghostNeighborOffsets[i+1] (exclusive).
    */
   Containers::Array< std::uint8_t, DeviceType, GlobalIndexType > sendBuffers;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGridSynchronizer.h>
