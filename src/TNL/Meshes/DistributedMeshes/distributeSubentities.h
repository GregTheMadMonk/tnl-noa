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

#include <TNL/Meshes/Geometry/getEntityCenter.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

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

   // 0. exchange vertex data to prepare getVertexOwner and later on synchronizeSparse
   DistributedMeshSynchronizer< DistributedMesh, 0 > synchronizer;
   synchronizer.initialize( mesh );

   auto getVertexOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      const GlobalIndexType global_idx = mesh.template getGlobalIndices< 0 >()[ local_idx ];
      return synchronizer.getEntityOwner( global_idx );
   };

   // find which rank owns all vertices of its local cells
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
      for( int i = 0; i < nproc; i++ )
         if( recvbuf[ i ] ) {
            rankOwningAllLocalCellSubvertices = i;
            break;
         }
   }
   if( rankOwningAllLocalCellSubvertices != 0 && rankOwningAllLocalCellSubvertices != nproc - 1 )
      throw std::runtime_error("Vertices are not distributed consistently. Shared vertices on the boundaries must be assigned "
                               "either to the highest or to the lowest rank. Thus, either the first or the last rank must "
                               "own all subvertices of its local cells.");

   auto getEntityOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      auto entity = mesh.getLocalMesh().template getEntity< Dimension >( local_idx );
      int owner = (rankOwningAllLocalCellSubvertices == 0) ? 0 : nproc;
      if( rankOwningAllLocalCellSubvertices == 0 ) {
         // this assumes that vertices at the boundaries were assigned to the subdomain with the lowest rank
         // (this is used in DistributedMeshTest for simplicitty)
         for( LocalIndexType v = 0; v < entity.template getSubentitiesCount< 0 >(); v++ ) {
            const GlobalIndexType gv = entity.template getSubentityIndex< 0 >( v );
            owner = TNL::max( owner, getVertexOwner( gv ) );
         }
      }
      else {
         // this assumes that vertices at the boundaries were assigned to the subdomain with the highest rank
         // (this is what tnl-decompose-mesh does)
         for( LocalIndexType v = 0; v < entity.template getSubentitiesCount< 0 >(); v++ ) {
            const GlobalIndexType gv = entity.template getSubentityIndex< 0 >( v );
            owner = TNL::min( owner, getVertexOwner( gv ) );
         }
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

   // 6. exchange local indices for ghost entities
   // We have to synchronize the vertex-entity superentity matrix, synchronization based
   // on the cell-entity subentity matrix is not general. For example, two subdomains can
   // have a common face, but no common cell, even when ghost_levels > 0. On the other
   // hand, if two subdomains have a common face, they have common all its subvertices,
   // so it is ensured that we send/receive indices for all ghost entities (with a rather
   // great redundancy).
   const auto sparseResult = synchronizer.synchronizeSparse( localMesh.template getSuperentitiesMatrix< 0, Dimension >() );
   const auto& rankOffsets = std::get< 0 >( sparseResult );
   const auto& rowPointers = std::get< 1 >( sparseResult );
   const auto& columnIndices = std::get< 2 >( sparseResult );

   // 7. set the global indices of our ghost entities
   localMesh.template forGhost< Dimension >( [&] ( GlobalIndexType entityIndex ) mutable {
      const int owner = getEntityOwner( entityIndex );
      for( LocalIndexType v = 0; v < localMesh.template getSubentitiesCount< Dimension, 0 >( entityIndex ); v++ ) {
         const GlobalIndexType vertex = localMesh.template getSubentityIndex< Dimension, 0 >( entityIndex, v );
         const int vertexOwner = getVertexOwner( vertex );
         if( vertexOwner == owner ) {
            const GlobalIndexType ghostOffset = vertex - synchronizer.getGhostOffsets()[ vertexOwner ];
            // global index = owner's local index + owner's offset
            const GlobalIndexType globalEntityIndex = columnIndices[ rowPointers[ rankOffsets[ vertexOwner ] + ghostOffset ] + v ] + globalOffsets[ owner ];
            mesh.template getGlobalIndices< Dimension >()[ entityIndex ] = globalEntityIndex;
            break;
         }
      }
   });

   // 8. reorder the entities to make sure that global indices are sorted
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
