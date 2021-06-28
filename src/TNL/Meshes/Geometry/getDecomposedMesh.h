#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>
#include <TNL/Meshes/Geometry/EntityDecomposer.h>

namespace TNL {
namespace Meshes {

// Polygon Mesh
template< typename ParentConfig >
struct TriangleConfig: public ParentConfig
{
   using CellTopology = Topologies::Triangle;
};

template< EntityDecomposerVersion EntityDecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< TriangleMesh >;
   using PointType = typename TriangleMesh::PointType;
   using GlobalIndexType = typename TriangleMesh::GlobalIndexType;
   using LocalIndexType = typename TriangleMesh::LocalIndexType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, EntityDecomposerVersion >;
   
   TriangleMesh outMesh;
   MeshBuilder meshBuilder;
   
   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 2 >();

   // Find the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      GlobalIndexType extraPointsCount, entitiesCount;
      std::tie( extraPointsCount, entitiesCount ) = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
      outPointsCount += extraPointsCount;
      outCellsCount += entitiesCount;
   }
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // Copy the points from inMesh to outMesh
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

   // Lambda for setting decomposed triangle in meshBuilder
   GlobalIndexType setCellsCount = 0;
   auto setDecomposedCellFunc = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
      auto & entitySeed = meshBuilder.getCellSeed( setCellsCount++ );
      entitySeed.setCornerId( 0, v0 );
      entitySeed.setCornerId( 1, v1 );
      entitySeed.setCornerId( 2, v2 );
   };

   // Decompose each cell
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      EntityDecomposer::decompose( cell, createPointFunc, setDecomposedCellFunc );
   }
   
   meshBuilder.build( outMesh );
   return outMesh;
}

// Polyhedral Mesh
template< typename ParentConfig >
struct TetrahedronConfig: public ParentConfig
{
   using CellTopology = Topologies::Tetrahedron;
};

template< EntityDecomposerVersion EntityDecomposerVersion_,
          EntityDecomposerVersion SubentityDecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;
   using PointType = typename TetrahedronMesh::PointType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polyhedron, EntityDecomposerVersion_, SubentityDecomposerVersion >;
   
   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // Find the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      GlobalIndexType extraPointsCount, entitiesCount;
      std::tie( extraPointsCount, entitiesCount ) = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
      outPointsCount += extraPointsCount;
      outCellsCount += entitiesCount;
   }
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // Copy the points from inMesh to outMesh
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
   auto setDecomposedCellFunc = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2, GlobalIndexType v3 ) {
      auto & entitySeed = meshBuilder.getCellSeed( setCellsCount++ );
      entitySeed.setCornerId( 0, v0 );
      entitySeed.setCornerId( 1, v1 );
      entitySeed.setCornerId( 2, v2 );
      entitySeed.setCornerId( 3, v3 );
   };

   // Decompose each cell
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      EntityDecomposer::decompose( cell, createPointFunc, setDecomposedCellFunc );
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

} // namespace Meshes
} // namespace TNL