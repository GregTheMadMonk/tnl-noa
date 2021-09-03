#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/Geometry/isPlanar.h>
#include <TNL/Meshes/Geometry/EntityDecomposer.h>

namespace TNL {
namespace Meshes {

// 3D Polygon Mesh
template< EntityDecomposerVersion EntityDecomposerVersion_,
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
   using PointType = typename PolygonMesh::PointType;
   using RealType = typename PolygonMesh::RealType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, EntityDecomposerVersion_ >;

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
         GlobalIndexType extraPointsCount, entitiesCount;
         std::tie( extraPointsCount, entitiesCount ) = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
         outPointsCount += extraPointsCount;
         outCellsCount += entitiesCount;
      }
   }
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // copy the points from inMesh to outMesh
   GlobalIndexType setPointsCount = 0;
   for( ; setPointsCount < inPointsCount; setPointsCount++ ) {
      meshBuilder.setPoint( setPointsCount, inMesh.getPoint( setPointsCount ) );
   }

   // Lambda for creating new points
   auto createPointFunc = [&] ( const PointType & point ) {
      const auto pointIdx = setPointsCount++;
      meshBuilder.setPoint( pointIdx, point );
      return pointIdx;
   };

   // Lambda for setting decomposed cells in meshBuilder
   GlobalIndexType setCellsCount = 0;
   auto setDecomposedCellFunc = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
      auto & entitySeed = meshBuilder.getCellSeed( setCellsCount++ );
      entitySeed.setCornersCount( 3 );
      entitySeed.setCornerId( 0, v0 );
      entitySeed.setCornerId( 1, v1 );
      entitySeed.setCornerId( 2, v2 );
   };

   // Decompose non-planar cells and copy the rest
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      if( isPlanar( inMesh, cell, precision ) ) { // Copy planar cells
         auto & cellSeed = meshBuilder.getCellSeed( setCellsCount++ );
         const auto verticesCount = cell.template getSubentitiesCount< 0 >();
         cellSeed.setCornersCount( verticesCount );
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            cellSeed.setCornerId( j, cell.template getSubentityIndex< 0 >( j ) );
         }
      }
      else { // Decompose non-planar cells
         EntityDecomposer::decompose( cell, createPointFunc, setDecomposedCellFunc );
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

// Polyhedral Mesh
template< EntityDecomposerVersion EntityDecomposerVersion_,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
Mesh< MeshConfig, Devices::Host >
getPlanarMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using PolyhedronMesh = Mesh< MeshConfig, Devices::Host >;
   using GlobalIndexType = typename PolyhedronMesh::GlobalIndexType;
   using LocalIndexType = typename PolyhedronMesh::LocalIndexType;
   using PointType = typename PolyhedronMesh::PointType;
   using RealType = typename PolyhedronMesh::RealType;
   using FaceMapArray = Containers::Array< std::pair< GlobalIndexType, GlobalIndexType >, Devices::Host, GlobalIndexType >;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, EntityDecomposerVersion_ >;

   constexpr RealType precision{ 1e-6 };

   PolyhedronMesh outMesh;
   MeshBuilder< PolyhedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inFacesCount = inMesh.template getEntitiesCount< 2 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   FaceMapArray faceMap( inFacesCount ); // Mapping of original face indeces to a group of decomposed face indices

   // Find the number of points and faces in the outMesh and setup faceMap
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outFacesCount = 0;
   for( GlobalIndexType i = 0; i < inFacesCount; i++ ) {
      const auto face = inMesh.template getEntity< 2 >( i );
      if( isPlanar( inMesh, face, precision ) ) {
         const auto startFaceIdx = outFacesCount;
         const auto endFaceIdx = ++outFacesCount; // Planar faces aren't decomposed
         faceMap[ i ] = { startFaceIdx, endFaceIdx };
      }
      else {
         GlobalIndexType extraPointsCount, entitiesCount;
         std::tie( extraPointsCount, entitiesCount ) = EntityDecomposer::getExtraPointsAndEntitiesCount( face );
         outPointsCount += extraPointsCount;
         const auto startFaceIdx = outFacesCount;
         outFacesCount += entitiesCount;
         const auto endFaceIdx = outFacesCount;
         faceMap[ i ] = { startFaceIdx, endFaceIdx };
      }
   }

   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setFacesCount( outFacesCount );
   meshBuilder.setCellsCount( inCellsCount ); // The number of cells stays the same

   // Copy the points from inMesh to outMesh
   GlobalIndexType setPointsCount = 0;
   for( ; setPointsCount < inPointsCount; setPointsCount++ ) {
      meshBuilder.setPoint( setPointsCount, inMesh.getPoint( setPointsCount ) );
   }

   // Set up cell seeds
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

   // set corners count for face seeds
   GlobalIndexType setFacesCount = 0;
   for( GlobalIndexType i = 0; i < inFacesCount; i++ ) {
      const auto & faceMapping = faceMap[ i ];
      const bool isPlanarRes = ( faceMapping.second - faceMapping.first ) == 1;
      if( isPlanarRes ) {
         const auto face = inMesh.template getEntity< 2 >( i );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         meshBuilder.setFaceCornersCount( setFacesCount++, verticesCount );
      }
      else {
         for( GlobalIndexType j = faceMapping.first; j < faceMapping.second; j++ ) {
            meshBuilder.setFaceCornersCount( setFacesCount++, 3 );
         }
      }
   }
   meshBuilder.initializeFaceSeeds();

   // Lambda for creating new points
   auto createPointFunc = [&] ( const PointType & point ) {
      const auto pointIdx = setPointsCount++;
      meshBuilder.setPoint( pointIdx, point );
      return pointIdx;
   };

   // Lambda for setting seed of decomposed triangle
   setFacesCount = 0;
   auto setDecomposedFaceFunc = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
      const GlobalIndexType faceId = setFacesCount++;
      auto seed = meshBuilder.getFaceSeed( faceId );
      seed.setCornerId( 0, v0 );
      seed.setCornerId( 1, v1 );
      seed.setCornerId( 2, v2 );
   };

   // Decompose non-planar faces and copy the rest
   for( GlobalIndexType i = 0; i < inFacesCount; i++ ) {
      const auto face = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = face.template getSubentitiesCount< 0 >();
      const auto & faceMapping = faceMap[ i ];
      const bool isPlanarRes = ( faceMapping.second - faceMapping.first ) == 1; // Face was planar if face maps only onto 1 face
      if( isPlanarRes ) { // Copy planar faces
         const GlobalIndexType faceId = setFacesCount++;
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            meshBuilder.getFaceSeed( faceId ).setCornerId( j, face.template getSubentityIndex< 0 >( j ) );
         }
      }
      else { // Decompose non-planar cells
         EntityDecomposer::decompose( face, createPointFunc, setDecomposedFaceFunc );
      }
   }

   faceMap.reset();
   meshBuilder.build( outMesh );
   return outMesh;
}

} // namespace Meshes
} // namespace TNL
