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

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename MeshFunction,
          // Mesh is used only for DistributedGrid specializations
          typename Mesh = typename MeshFunction::MeshType >
class DistributedMeshSynchronizer
{
public:
   // TODO: generalize
   static constexpr int EntityDimension = MeshFunction::getEntitiesDimension();
   static_assert( EntityDimension == 0 || EntityDimension == MeshFunction::getMeshDimension(),
                  "Synchronization for entities of the specified dimension is not implemented yet." );

   using RealType = typename MeshFunction::RealType;
   using DeviceType = typename MeshFunction::DeviceType;
   using IndexType = typename MeshFunction::IndexType;
   using CommunicatorType = Communicators::MpiCommunicator;

   DistributedMeshSynchronizer() = default;

   void initialize( const Mesh& mesh )
   {
      static_assert( std::is_same< typename Mesh::CommunicatorType, CommunicatorType >::value,
                     "Mesh::CommunnicatorType does not match" );
      localEntitiesCount = mesh.getLocalMesh().template getEntitiesCount< EntityDimension >();

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
      const IndexType ownStart = mesh.template getGlobalIndices< EntityDimension >()[ 0 ];
      Containers::Array< IndexType, DeviceType > offsets( nproc );
      {
         Containers::Array< IndexType, DeviceType > sendbuf( nproc );
         sendbuf.setValue( ownStart );
         CommunicatorType::Alltoall( sendbuf.getData(), 1,
                                     offsets.getData(), 1,
                                     group );
      }

      auto getOwner = [&] ( IndexType idx )
      {
         // TODO: is there a more efficient way?
         for( int i = 0; i < nproc - 1; i++ )
            if( offsets[ i ] <= idx && idx < offsets[ i + 1 ] )
               return i;
         return nproc - 1;
      };

      // TODO: initialization of the distributed mesh should set the ranges for
      //       local and ghost entities so we can iterate just over the ghost
      //       entities
      // FIXME: the local vertices are not contiguous !!!
      // count local ghost entities for each rank
      Containers::Array< IndexType, DeviceType > localGhostCounts( nproc );
      localGhostCounts.setValue( 0 );
      IndexType firstGhost = 0;
      bool ghost_found = false;
      for( IndexType local_idx = 0; local_idx < localEntitiesCount; local_idx++ ) {
         const IndexType global_idx = mesh.template getGlobalIndices< EntityDimension >()[ local_idx ];
         const bool isNeighbor = mesh.getLocalMesh().template isGhostEntity< EntityDimension >( local_idx );
         if( ! isNeighbor ) {
            // TODO: this is just for testing/debugging
            if( ghost_found )
               std::cerr << "ERROR: global indices of local entities are not contiguous (local index " << local_idx << " is after a ghost entity has been found)" << std::endl;
            ++firstGhost;
         }
         else {
            const int owner = getOwner( global_idx );
            if( owner != rank ) {
               ++localGhostCounts[ owner ];
               ghost_found = true;
            }
         }
      }

      // exchange the ghost counts
      ghostEntitiesCounts.setDimensions( nproc, nproc );
      {
         Matrices::DenseMatrix< IndexType, Devices::Host, int > sendbuf;
         sendbuf.setDimensions( nproc, nproc );
         // copy the local ghost counts into all rows of the sendbuf
         for( int j = 0; j < nproc; j++ )
         for( int i = 0; i < nproc; i++ )
            sendbuf.setElement( j, i, localGhostCounts.getElement( i ) );
         CommunicatorType::Alltoall( &sendbuf(0, 0), nproc,
                                     &ghostEntitiesCounts(0, 0), nproc,
                                     group );
      }

      // allocate ghost offsets
      ghostOffsets.setSize( nproc );

      // set ghost neighbor offsets
      ghostNeighborOffsets.setSize( nproc + 1 );
      ghostNeighborOffsets[ 0 ] = 0;
      for( int i = 0; i < nproc; i++ )
         ghostNeighborOffsets[ i + 1 ] = ghostNeighborOffsets[ i ] + ghostEntitiesCounts( i, rank );

      // allocate ghost neighbors and send buffers
      ghostNeighbors.setSize( ghostNeighborOffsets[ nproc ] );
      sendBuffers.setSize( ghostNeighborOffsets[ nproc ] );

      // send indices of ghost entities - set them as ghost neighbors on the
      // target rank
      {
         std::vector< typename CommunicatorType::Request > requests;

         // send our ghost indices to the neighboring ranks
         for( int i = 0; i < nproc; i++ ) {
            if( ghostEntitiesCounts( rank, i ) > 0 ) {
               requests.push_back( CommunicatorType::ISend(
                        mesh.template getGlobalIndices< EntityDimension >().getData() + firstGhost,
                        ghostEntitiesCounts( rank, i ),
                        i, 0, group ) );
               // update ghost offsets
               ghostOffsets[ i ] = firstGhost;
               firstGhost += ghostEntitiesCounts( rank, i );
            }
            else
               ghostOffsets[ i ] = 0;
         }

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

#if 0
      CommunicatorType::Barrier();
      for( int i = 0; i < nproc; i++ ) {
         if( i == rank ) {
            std::cout << "rank = " << rank << "\n";
//            std::cout << "offsets = " << offsets << std::endl;
            std::cout << "local ghost counts = " << localGhostCounts << "\n";
            std::cout << "ghost entities counts matrix:\n" << ghostEntitiesCounts;
            std::cout << "global indices = " << mesh.template getGlobalIndices< EntityDimension >() << "\n";
            std::cout << "ghost offsets = " << ghostOffsets << "\n";
            std::cout << "ghost neighbors = " << ghostNeighbors << "\n";
            std::cout.flush();
         }
         CommunicatorType::Barrier();
      }
#endif
   }

   void synchronize( MeshFunction& function )
   {
      TNL_ASSERT_EQ( function.getData().getSize(), localEntitiesCount,
                     "The mesh function does not have the expected size." );

      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      #ifdef HAVE_CUDA
      if( std::is_same< DeviceType, Devices::Cuda >::value )
         cudaSetDevice(gpu_id);
      #endif

      const int rank = CommunicatorType::GetRank( group );
      const int nproc = CommunicatorType::GetSize( group );

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

      for( int i = 0; i < nproc; i++ ) {
         if( ghostEntitiesCounts( i, rank ) > 0 ) {
            const IndexType offset = ghostNeighborOffsets[ i ];
            // copy data to send buffers
            for( IndexType k = 0; k < ghostEntitiesCounts( i, rank ); k++ )
               sendBuffers[ offset + k ] = function.getData()[ ghostNeighbors[ offset + k ] ];

            // issue async send operation
            requests.push_back( CommunicatorType::ISend(
                     sendBuffers.getData() + ghostNeighborOffsets[ i ],
                     ghostEntitiesCounts( i, rank ),
                     i, 0, group ) );
         }
      }

      // wait for all communications to finish
      CommunicatorType::WaitAll( requests.data(), requests.size() );
   }

protected:
   // count of local entities (including ghosts) - used only for asserts in the
   // synchronize method
   IndexType localEntitiesCount = 0;

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
   Matrices::DenseMatrix< IndexType, Devices::Host, int > ghostEntitiesCounts;

   /**
    * Ghost offsets: the i-th value is the local index of the first ghost
    * entity owned by the i-th rank. All ghost entities owned by the i-th
    * rank are assumed to be indexed contiguously in the local mesh.
    */
   Containers::Array< IndexType, DeviceType, IndexType > ghostOffsets;

   /**
    * Ghost neighbor offsets: array of size nproc + 1 where the i-th value is
    * the offset of ghost neighbor indices requested by the i-th rank. The last
    * value is the size of the ghostNeighbors and sendBuffers arrays (see
    * below).
    */
   Containers::Array< IndexType, Devices::Host, int > ghostNeighborOffsets;

   /**
    * Ghost neighbor indices: array containing local indices of the entities
    * which are ghosts on other ranks. The indices requested by the i-th rank
    * are in the range starting at ghostNeighborOffsets[i] (inclusive) and
    * ending at ghostNeighborOffsets[i+1] (exclusive). These indices are used
    * for copying the mesh function values into the sendBuffers array. Note that
    * ghost neighbor indices cannot be made contiguous in general so we need the
    * send buffers.
    */
   Containers::Vector< IndexType, DeviceType, IndexType > ghostNeighbors;

   /**
    * Send buffers: array for buffering the mesh function values which will be
    * sent to other ranks. The send buffer for the i-th rank is the part of the
    * array starting at index ghostNeighborOffsets[i] (inclusive) and ending at
    * index ghostNeighborOffsets[i+1] (exclusive).
    */
   Containers::Array< RealType, DeviceType, IndexType > sendBuffers;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGridSynchronizer.h>
