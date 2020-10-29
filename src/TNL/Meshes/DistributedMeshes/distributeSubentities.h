/***************************************************************************
                          distributeSubentities.h  -  description
                             -------------------
    begin                : July 13, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <numeric>   // std::iota
#include <atomic>

#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/MeshDetails/layers/EntityTags/Traits.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename CommunicatorType, typename GlobalIndexType >
auto
exchangeGhostEntitySeeds( typename CommunicatorType::CommunicationGroup group,
                          const std::vector< std::vector< GlobalIndexType > >& seeds_vertex_indices,
                          const std::vector< std::vector< GlobalIndexType > >& seeds_entity_offsets )
{
   const int rank = CommunicatorType::GetRank( group );
   const int nproc = CommunicatorType::GetSize( group );

   // exchange sizes of the arrays
   Containers::Array< GlobalIndexType, Devices::Host, int > sizes_vertex_indices( nproc ), sizes_entity_offsets( nproc );
   {
      Containers::Array< GlobalIndexType, Devices::Host, int > sendbuf_indices( nproc ), sendbuf_offsets( nproc );
      for( int i = 0; i < nproc; i++ ) {
         sendbuf_indices[ i ] = seeds_vertex_indices[ i ].size();
         sendbuf_offsets[ i ] = seeds_entity_offsets[ i ].size();
      }
      CommunicatorType::Alltoall( sendbuf_indices.getData(), 1,
                                  sizes_vertex_indices.getData(), 1,
                                  group );
      CommunicatorType::Alltoall( sendbuf_offsets.getData(), 1,
                                  sizes_entity_offsets.getData(), 1,
                                  group );
   }

   // allocate arrays for the results
   std::vector< std::vector< GlobalIndexType > > foreign_seeds_vertex_indices, foreign_seeds_entity_offsets;
   foreign_seeds_vertex_indices.resize( nproc );
   foreign_seeds_entity_offsets.resize( nproc );
   for( int i = 0; i < nproc; i++ ) {
      foreign_seeds_vertex_indices[ i ].resize( sizes_vertex_indices[ i ] );
      foreign_seeds_entity_offsets[ i ].resize( sizes_entity_offsets[ i ] );
   }

   // buffer for asynchronous communication requests
   std::vector< typename CommunicatorType::Request > requests;

   // issue all async receive operations
   for( int j = 0; j < nproc; j++ ) {
      if( j == rank )
          continue;
      requests.push_back( CommunicatorType::IRecv(
               foreign_seeds_vertex_indices[ j ].data(),
               foreign_seeds_vertex_indices[ j ].size(),
               j, 0, group ) );
      requests.push_back( CommunicatorType::IRecv(
               foreign_seeds_entity_offsets[ j ].data(),
               foreign_seeds_entity_offsets[ j ].size(),
               j, 1, group ) );
   }

   // issue all async send operations
   for( int i = 0; i < nproc; i++ ) {
      if( i == rank )
          continue;
      requests.push_back( CommunicatorType::ISend(
               seeds_vertex_indices[ i ].data(),
               seeds_vertex_indices[ i ].size(),
               i, 0, group ) );
      requests.push_back( CommunicatorType::ISend(
               seeds_entity_offsets[ i ].data(),
               seeds_entity_offsets[ i ].size(),
               i, 1, group ) );
   }

   // wait for all communications to finish
   CommunicatorType::WaitAll( requests.data(), requests.size() );

   return std::make_tuple( foreign_seeds_vertex_indices, foreign_seeds_entity_offsets );
}

template< typename CommunicatorType, typename GlobalIndexType >
auto
exchangeGhostIndices( typename CommunicatorType::CommunicationGroup group,
                      const std::vector< std::vector< GlobalIndexType > >& foreign_ghost_indices,
                      const std::vector< std::vector< GlobalIndexType > >& seeds_local_indices )
{
   const int rank = CommunicatorType::GetRank( group );
   const int nproc = CommunicatorType::GetSize( group );

   // allocate arrays for the results
   std::vector< std::vector< GlobalIndexType > > ghost_indices;
   ghost_indices.resize( nproc );
   for( int i = 0; i < nproc; i++ )
      ghost_indices[ i ].resize( seeds_local_indices[ i ].size() );

   // buffer for asynchronous communication requests
   std::vector< typename CommunicatorType::Request > requests;

   // issue all async receive operations
   for( int j = 0; j < nproc; j++ ) {
      if( j == rank )
          continue;
      requests.push_back( CommunicatorType::IRecv(
               ghost_indices[ j ].data(),
               ghost_indices[ j ].size(),
               j, 0, group ) );
   }

   // issue all async send operations
   for( int i = 0; i < nproc; i++ ) {
      if( i == rank )
          continue;
      requests.push_back( CommunicatorType::ISend(
               foreign_ghost_indices[ i ].data(),
               foreign_ghost_indices[ i ].size(),
               i, 0, group ) );
   }

   // wait for all communications to finish
   CommunicatorType::WaitAll( requests.data(), requests.size() );

   return ghost_indices;
}

// FIXME: This algorithm works only when min-common-vertices == 1, i.e. we have
// the full information about neighbors of ghosts on the overlap. Otherwise,
// depending on how the mesh was decomposed, we might end up with errors like
//    vertex with gid=XXX received from rank X was not found on the local mesh for rank Y (global offset = YYY)
// The problem is with the getEntityOwner function, which assumes that it knows
// everything about the neighbors of the entity based on its subvertices.
//
// FIXME: This algorithm may distribute entities in such a way that some rank
// owns an entity on the interface between two (other) subdomains, but the
// neighbor cell of the entity is a ghost.
template< int Dimension, typename DistributedMesh >
void
distributeSubentities( DistributedMesh& mesh )
{
   using DeviceType = typename DistributedMesh::DeviceType;
   using GlobalIndexType = typename DistributedMesh::GlobalIndexType;
   using LocalIndexType = typename DistributedMesh::LocalIndexType;
   using LocalMesh = typename DistributedMesh::MeshType;
   using CommunicatorType = typename DistributedMesh::CommunicatorType;

   static_assert( ! std::is_same< DeviceType, Devices::Cuda >::value,
                  "this method can be called only for host meshes" );
   static_assert( 0 < Dimension && Dimension < DistributedMesh::getMeshDimension(),
                  "Vertices and cells cannot be distributed using this method." );
   if( mesh.getGhostLevels() <= 0 )
      throw std::logic_error( "There are no ghost levels on the distributed mesh." );

   const int rank = CommunicatorType::GetRank( mesh.getCommunicationGroup() );
   const int nproc = CommunicatorType::GetSize( mesh.getCommunicationGroup() );

   // 0. exchange vertex data to prepare getVertexOwner for use in getEntityOwner
   DistributedMeshSynchronizer< DistributedMesh, 0 > synchronizer;
   synchronizer.initialize( mesh );

   auto getVertexOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      const GlobalIndexType global_idx = mesh.template getGlobalIndices< 0 >()[ local_idx ];
      return synchronizer.getEntityOwner( global_idx );
   };

   // find which rank owns all vertices of its local cells
   // (this is not unique, there might be more such subdomains (which are not connected),
   // so we assume it's either 0 or nproc-1)
   int rankOwningAllLocalCellSubvertices = nproc;
   {
      std::atomic<bool> its_us( true );
      mesh.getLocalMesh().template forLocal< DistributedMesh::getMeshDimension() >( [&] ( GlobalIndexType i ) mutable {
         for( LocalIndexType v = 0; v < mesh.getLocalMesh().template getSubentitiesCount< DistributedMesh::getMeshDimension(), 0 >( i ); v++ ) {
            const GlobalIndexType gv = mesh.getLocalMesh().template getSubentityIndex< DistributedMesh::getMeshDimension(), 0 >( i, v );
            if( getVertexOwner( gv ) != rank )
               its_us = false;
         }
      });
      Containers::Array< bool, Devices::Host, int > recvbuf( nproc ), sendbuf( nproc );
      sendbuf.setValue( its_us );
      CommunicatorType::Alltoall( sendbuf.getData(), 1,
                                  recvbuf.getData(), 1,
                                  mesh.getCommunicationGroup() );
      if( recvbuf[ 0 ] )
         rankOwningAllLocalCellSubvertices = 0;
      else if( recvbuf[ nproc - 1 ] )
         rankOwningAllLocalCellSubvertices = nproc - 1;
      else
         throw std::runtime_error("Vertices are not distributed consistently. Shared vertices on the boundaries must be assigned "
                                  "either to the highest or to the lowest rank. Thus, either the first or the last rank must "
                                  "own all subvertices of its local cells.");
   }

   auto getEntityOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      auto entity = mesh.getLocalMesh().template getEntity< Dimension >( local_idx );

      // if all neighbor cells are local, we are the owner
      bool all_neighbor_cells_local = true;
      for( LocalIndexType k = 0; k < entity.template getSuperentitiesCount< DistributedMesh::getMeshDimension() >(); k++ ) {
         const GlobalIndexType gk = entity.template getSuperentityIndex< DistributedMesh::getMeshDimension() >( k );
         if( mesh.getLocalMesh().template isGhostEntity< DistributedMesh::getMeshDimension() >( gk ) ) {
            all_neighbor_cells_local = false;
            break;
         }
      }
      if( all_neighbor_cells_local )
         return rank;

      int owner = (rankOwningAllLocalCellSubvertices == 0) ? 0 : nproc;
      for( LocalIndexType v = 0; v < entity.template getSubentitiesCount< 0 >(); v++ ) {
         const GlobalIndexType gv = entity.template getSubentityIndex< 0 >( v );
         if( rankOwningAllLocalCellSubvertices == 0 )
            // this assumes that vertices at the boundaries were assigned to the subdomain with the lowest rank
            // (this is used in DistributedMeshTest for simplicitty)
            owner = TNL::max( owner, getVertexOwner( gv ) );
         else
            // this assumes that vertices at the boundaries were assigned to the subdomain with the highest rank
            // (this is what tnl-decompose-mesh does)
            owner = TNL::min( owner, getVertexOwner( gv ) );
      }
      return owner;
   };

   // 1. identify local entities, set the GhostEntity tag on others
   LocalMesh& localMesh = mesh.getLocalMesh();
   localMesh.template forAll< Dimension >( [&] ( GlobalIndexType i ) mutable {
      if( getEntityOwner( i ) != rank )
         localMesh.template addEntityTag< Dimension >( i, EntityTags::GhostEntity );
   });

   // 2. reorder the entities to make sure that all ghost entities are after local entities
   // TODO: it would be nice if the mesh initializer could do this
   {
      // count local entities
      GlobalIndexType localEntitiesCount = 0;
      for( GlobalIndexType i = 0; i < localMesh.template getEntitiesCount< Dimension >(); i++ )
         if( ! localMesh.template isGhostEntity< Dimension >( i ) )
            ++localEntitiesCount;

      // create the permutation
      typename LocalMesh::GlobalIndexArray perm, iperm;
      perm.setSize( localMesh.template getEntitiesCount< Dimension >() );
      iperm.setSize( localMesh.template getEntitiesCount< Dimension >() );
      GlobalIndexType localsCount = 0;
      GlobalIndexType ghostsCount = 0;
      for( GlobalIndexType j = 0; j < iperm.getSize(); j++ ) {
         if( localMesh.template isGhostEntity< Dimension >( j ) ) {
            iperm[ j ] = localEntitiesCount + ghostsCount;
            perm[ localEntitiesCount + ghostsCount ] = j;
            ++ghostsCount;
         }
         else {
            iperm[ j ] = localsCount;
            perm[ localsCount ] = j;
            ++localsCount;
         }
      }

      // reorder the local mesh
      localMesh.template reorderEntities< Dimension >( perm, iperm );
   }

   // 3. update entity tags layer (this is actually done as part of the mesh reordering)
   //localMesh.template updateEntityTagsLayer< Dimension >();

   // 4. exchange the counts of local entities between ranks, compute offsets for global indices
   Containers::Vector< GlobalIndexType, Devices::Host, int > globalOffsets( nproc );
   {
      Containers::Array< GlobalIndexType, Devices::Host, int > sendbuf( nproc );
      sendbuf.setValue( localMesh.template getGhostEntitiesOffset< Dimension >() );
      CommunicatorType::Alltoall( sendbuf.getData(), 1,
                                  globalOffsets.getData(), 1,
                                  mesh.getCommunicationGroup() );
   }
   globalOffsets.template scan< Algorithms::ScanType::Exclusive >();

   // 5. assign global indices to the local entities
   mesh.template getGlobalIndices< Dimension >().setSize( localMesh.template getEntitiesCount< Dimension >() );
   localMesh.template forLocal< Dimension >( [&] ( GlobalIndexType i ) mutable {
      mesh.template getGlobalIndices< Dimension >()[ i ] = globalOffsets[ rank ] + i;
   });

   // Now for each ghost entity, we will take the global indices of its subvertices and
   // send them to the owner of the entity. The owner will scan its vertex-entity
   // superentity matrix, find the entity which has the received vertex indices and send
   // the global entity index back to the inquirer.
   // Note that we have to synchronize based on the vertex-entity superentity matrix,
   // because synchronization based on the cell-entity subentity matrix would not be
   // general. For example, two subdomains can have a common face, but no common cell,
   // even when ghost_levels > 0. On the other hand, if two subdomains have a common face,
   // they have common all its subvertices.

   // 6. build seeds for ghost entities
   std::vector< std::vector< GlobalIndexType > > seeds_vertex_indices, seeds_entity_offsets, seeds_local_indices;
   seeds_vertex_indices.resize( nproc );
   seeds_entity_offsets.resize( nproc );
   seeds_local_indices.resize( nproc );
   for( GlobalIndexType entity_index = localMesh.template getGhostEntitiesOffset< Dimension >();
        entity_index < localMesh.template getEntitiesCount< Dimension >();
        entity_index++ )
   {
      const int owner = getEntityOwner( entity_index );
      for( LocalIndexType v = 0; v < localMesh.template getSubentitiesCount< Dimension, 0 >( entity_index ); v++ ) {
         const GlobalIndexType local_index = localMesh.template getSubentityIndex< Dimension, 0 >( entity_index, v );
         const GlobalIndexType global_index = mesh.template getGlobalIndices< 0 >()[ local_index ];
         seeds_vertex_indices[ owner ].push_back( global_index );
      }
      seeds_entity_offsets[ owner ].push_back( seeds_vertex_indices[ owner ].size() );
      // record the corresponding local index for later use
      seeds_local_indices[ owner ].push_back( entity_index );
   }

   // 7. exchange seeds for ghost entities
   const auto foreign_seeds = exchangeGhostEntitySeeds< CommunicatorType >( mesh.getCommunicationGroup(), seeds_vertex_indices, seeds_entity_offsets );
   const auto& foreign_seeds_vertex_indices = std::get< 0 >( foreign_seeds );
   const auto& foreign_seeds_entity_offsets = std::get< 1 >( foreign_seeds );

//   std::stringstream msg;
//   msg << "rank " << rank << ":\n";
//   for( int i = 0; i < nproc; i++ ) {
//      msg << "- from rank " << i << ":\n";
//      msg << "\tindices: ";
//      for( auto j : foreign_seeds_vertex_indices[i] )
//         msg << j << " ";
//      msg << "\n\toffsets: ";
//      for( auto j : foreign_seeds_entity_offsets[i] )
//         msg << j << " ";
//      msg << "\n";
//   }
//   std::cout << msg.str();

   // 8. determine global indices for the received seeds
   std::vector< std::vector< GlobalIndexType > > foreign_ghost_indices;
   foreign_ghost_indices.resize( nproc );
   for( int i = 0; i < nproc; i++ )
      foreign_ghost_indices[ i ].resize( foreign_seeds_entity_offsets[ i ].size() );
   Algorithms::ParallelFor< Devices::Host >::exec( 0, nproc, [&] ( int i ) {
      GlobalIndexType vertexOffset = 0;
      // loop over all foreign ghost entities
      for( std::size_t entityIndex = 0; entityIndex < foreign_seeds_entity_offsets[ i ].size(); entityIndex++ ) {
         // data structure for common indices
         std::set< GlobalIndexType > common_indices;

         // loop over all subvertices of the entity
         while( vertexOffset < foreign_seeds_entity_offsets[ i ][ entityIndex ] ) {
            const GlobalIndexType vertex = foreign_seeds_vertex_indices[ i ][ vertexOffset++ ];
            GlobalIndexType localIndex = 0;
            if( vertex >= synchronizer.getGlobalOffsets()[ rank ]
                && vertex < synchronizer.getGlobalOffsets()[ rank ] + localMesh.template getGhostEntitiesOffset< 0 >() )
            {
               // subtract offset to get local index
               localIndex = vertex - synchronizer.getGlobalOffsets()[ rank ];
            }
            else {
               // we must go through the ghost entities
               for( GlobalIndexType g = localMesh.template getGhostEntitiesOffset< 0 >();
                    g < localMesh.template getEntitiesCount< 0 >();
                    g++ )
                  if( vertex == mesh.template getGlobalIndices< 0 >()[ g ] ) {
                     localIndex = g;
                     break;
                  }
               if( localIndex == 0 )
                  throw std::runtime_error( "vertex with gid=" + std::to_string(vertex) + " received from rank "
                                          + std::to_string(i) + " was not found on the local mesh for rank " + std::to_string(rank)
                                          + " (global offset = " + std::to_string(synchronizer.getGlobalOffsets()[ rank ]) + ")" );
            }

            // collect superentities of this vertex
            std::set< GlobalIndexType > superentities;
            for( LocalIndexType e = 0; e < localMesh.template getSuperentitiesCount< 0, Dimension >( localIndex ); e++ ) {
               const GlobalIndexType entity = localMesh.template getSuperentityIndex< 0, Dimension >( localIndex, e );
               superentities.insert( entity );
            }

            // initialize or intersect
            if( common_indices.empty() )
               common_indices = superentities;
            else
               // remove indices which are not in the current superentities set
               for( auto it = common_indices.begin(); it != common_indices.end(); ) {
                  if( superentities.count( *it ) == 0 )
                     it = common_indices.erase(it);
                  else
                     ++it;
               }
         }

         if( common_indices.size() != 1 ) {
            std::stringstream msg;
            msg << "expected exactly 1 common index, but the algorithm found these common indices: ";
            for( auto i : common_indices )
               msg << i << " ";
            msg << "\nDebug info: rank " << rank << ", entityIndex = " << entityIndex << ", received from rank " << i;
            throw std::runtime_error( msg.str() );
         }

         const GlobalIndexType local_index = *common_indices.begin();
         if( getEntityOwner( local_index ) != rank )
            throw std::runtime_error( "rank " + std::to_string(rank) + " does not own the entity which was left common: " + std::to_string(local_index) );

         // assign global index
         foreign_ghost_indices[ i ][ entityIndex ] = mesh.template getGlobalIndices< Dimension >()[ local_index ];
      }
   });

   // 9. exchange global ghost indices
   const auto ghost_indices = exchangeGhostIndices< CommunicatorType >( mesh.getCommunicationGroup(), foreign_ghost_indices, seeds_local_indices );

//   std::stringstream msg;
//   msg << "rank " << rank << ":\n";
//   for( int i = 0; i < nproc; i++ ) {
//      msg << "- from rank " << i << ":\n";
//      msg << "\tghost indices: ";
//      for( auto j : ghost_indices[i] )
//         msg << j << " ";
//      msg << "\n";
//   }
//   std::cout << msg.str();

   // 10. set the global indices of our ghost entities
   for( int i = 0; i < nproc; i++ ) {
      for( std::size_t g = 0; g < ghost_indices[ i ].size(); g++ )
         mesh.template getGlobalIndices< Dimension >()[ seeds_local_indices[ i ][ g ] ] = ghost_indices[ i ][ g ];
   }

   // 11. reorder the entities to make sure that global indices are sorted
   {
      // prepare vector with an identity permutation
      std::vector< GlobalIndexType > permutation( localMesh.template getEntitiesCount< Dimension >() );
      std::iota( permutation.begin(), permutation.end(), (GlobalIndexType) 0 );

      // sort the subarray corresponding to ghost entities by the global index
      std::stable_sort( permutation.begin() + mesh.getLocalMesh().template getGhostEntitiesOffset< Dimension >(),
                        permutation.end(),
                        [&mesh](auto& left, auto& right) {
         return mesh.template getGlobalIndices< Dimension >()[ left ] < mesh.template getGlobalIndices< Dimension >()[ right ];
      });

      // copy the permutation into TNL array and invert
      typename LocalMesh::GlobalIndexArray perm, iperm;
      perm.setSize( localMesh.template getEntitiesCount< Dimension >() );
      iperm.setSize( localMesh.template getEntitiesCount< Dimension >() );
      for( GlobalIndexType i = 0; i < localMesh.template getEntitiesCount< Dimension >(); i++ ) {
         perm[ i ] = permutation[ i ];
         iperm[ perm[ i ] ] = i;
      }

      // reorder the mesh
      mesh.template reorderEntities< Dimension >( perm, iperm );
   }
}

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
