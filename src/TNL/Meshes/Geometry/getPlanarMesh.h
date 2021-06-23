#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/Geometry/isPlanar.h>
#include <TNL/Meshes/Geometry/PolygonDecomposer.h>

namespace TNL {
namespace Meshes {

// 3D Polygon Mesh
template< PolygonDecomposerVersion version,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true,
          std::enable_if_t< MeshConfig::spaceDimension == 3, bool > = true >
Mesh< MeshConfig, Devices::Host >
getPlanarMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using PolygonMesh = Mesh< MeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< PolygonMesh >;
   using GlobalIndexType = typename PolygonMesh::GlobalIndexType;
   using LocalIndexType = typename PolygonMesh::LocalIndexType;
   using RealType = typename PolygonMesh::RealType;
   using PolygonDecomposer = PolygonDecomposer< MeshConfig, version >;

   constexpr RealType precision{ 1e-6 };

   PolygonMesh outMesh;
   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 2 >();

   // find the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outCellsCount = 0;

   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      if( isPlanar( inMesh, cell, precision ) ) { // Cell is not decomposed
         outCellsCount++;
      }
      else { // Cell is decomposed
         outPointsCount += PolygonDecomposer::getExtraPointsCount( cell );
         outCellsCount += PolygonDecomposer::getEntitiesCount( cell );
      }
   }

   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // copy the points from inMesh to outMesh
   GlobalIndexType setPointsCount = 0;
   for( ; setPointsCount < inPointsCount; setPointsCount++ ) {
      meshBuilder.setPoint( setPointsCount, inMesh.getPoint( setPointsCount ) );
   }

   auto cellSeedGetter = [&] ( const GlobalIndexType i ) -> auto& { return meshBuilder.getCellSeed( i ); };
   for( GlobalIndexType i = 0, setCellsCount = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      if( isPlanar( inMesh, cell, precision ) ) { // Copy planar cells
         auto & cellSeed = meshBuilder.getCellSeed( setCellsCount++ );
         const auto verticesCount = cell.template getSubentitiesCount< 0 >();
         cellSeed.setCornersCount( verticesCount );
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            cellSeed.setCornerId( j, cell.template getSubentityIndex< 0 >( j ) );
         }
      }
      else { // decompose non-planar cells
         PolygonDecomposer::decompose( meshBuilder,
                                       cellSeedGetter,
                                       setPointsCount,
                                       setCellsCount,
                                       cell );
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

// Polyhedral Mesh
template< PolygonDecomposerVersion version,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
Mesh< MeshConfig, Devices::Host >
getPlanarMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using PolyhedronMesh = Mesh< MeshConfig, Devices::Host >;
   using GlobalIndexType = typename PolyhedronMesh::GlobalIndexType;
   using LocalIndexType = typename PolyhedronMesh::LocalIndexType;
   using RealType = typename PolyhedronMesh::RealType;
   using FaceMapArray = Containers::Array< std::pair< GlobalIndexType, GlobalIndexType >, Devices::Host, GlobalIndexType >;
   using PolygonDecomposer = PolygonDecomposer< MeshConfig, version >;

   constexpr RealType precision{ 1e-6 };

   PolyhedronMesh outMesh;
   MeshBuilder< PolyhedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inFacesCount = inMesh.template getEntitiesCount< 2 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   FaceMapArray faceMap( inFacesCount );

   // find the number of points and faces in the outMesh
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outFacesCount = 0;
   for( GlobalIndexType i = 0; i < inFacesCount; i++ ) {
      const auto face = inMesh.template getEntity< 2 >( i );
      if( isPlanar( inMesh, face, precision ) ) { //
         const auto startFaceIdx = outFacesCount;
         const auto endFaceIdx = ++outFacesCount;
         faceMap[ i ] = { startFaceIdx, endFaceIdx };
      }
      else {
         outPointsCount += PolygonDecomposer::getExtraPointsCount( face );
         const auto startFaceIdx = outFacesCount;
         outFacesCount += PolygonDecomposer::getEntitiesCount( face );
         const auto endFaceIdx = outFacesCount;
         faceMap[ i ] = { startFaceIdx, endFaceIdx };
      }
   }

   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setFacesCount( outFacesCount );
   meshBuilder.setCellsCount( inCellsCount ); // The number of cells stays the same

   // copy the points from inMesh to outMesh
   GlobalIndexType setPointsCount = 0;
   for( ; setPointsCount < inPointsCount; setPointsCount++ ) {
      meshBuilder.setPoint( setPointsCount, inMesh.getPoint( setPointsCount ) );
   }

   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const LocalIndexType cellFacesCount = cell.template getSubentitiesCount< 2 >();

      // Find the number of faces in the output cell
      LocalIndexType cellSeedFacesCount = 0;
      for( LocalIndexType j = 0; j < cellFacesCount; j++ ) {
         const GlobalIndexType faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto & faceMapping = faceMap[ faceIdx ];
         cellSeedFacesCount += faceMapping.second - faceMapping.first;
      }

      // Set up the cell seed
      auto & cellSeed = meshBuilder.getCellSeed( i );
      cellSeed.setCornersCount( cellSeedFacesCount );
      for( LocalIndexType j = 0, faceSetIdx = 0; j < cellFacesCount; j++ ) {
         const GlobalIndexType faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto & faceMapping = faceMap[ faceIdx ];
         for( GlobalIndexType k = faceMapping.first; k < faceMapping.second; k++ ) {
            cellSeed.setCornerId( faceSetIdx++, k );
         }
      }
   }

   auto faceSeedGetter = [&] ( const GlobalIndexType i ) -> auto& { return meshBuilder.getFaceSeed( i ); };
   for( GlobalIndexType i = 0, setFacesCount = 0; i < inFacesCount; i++ ) {
      const auto face = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = face.template getSubentitiesCount< 0 >();
      const auto & faceMapping = faceMap[ i ];
      const bool isPlanarRes = ( faceMapping.second - faceMapping.first ) == 1; // face was planar if face maps only onto 1 face
      if( isPlanarRes ) { // Copy planar faces
         auto & faceSeed = meshBuilder.getFaceSeed( setFacesCount++ );
         faceSeed.setCornersCount( verticesCount );
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            faceSeed.setCornerId( j, face.template getSubentityIndex< 0 >( j ) );
         }
      }
      else { // decompose non-planar cells
         PolygonDecomposer::decompose( meshBuilder,
                                       faceSeedGetter,
                                       setPointsCount,
                                       setFacesCount,
                                       face );
      }
   }

   faceMap.reset();
   meshBuilder.build( outMesh );
   return outMesh;
}

} // namespace Meshes
} // namespace TNL
