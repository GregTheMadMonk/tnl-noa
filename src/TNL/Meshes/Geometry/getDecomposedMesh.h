#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/Geometry/EntityDecomposer.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/scan.h>

namespace TNL {
namespace Meshes {

// Polygon Mesh
template< typename ParentConfig >
struct TriangleConfig: public ParentConfig
{
   using CellTopology = Topologies::Triangle;
};

template< EntityDecomposerVersion DecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto // returns MeshBuilder
decomposeMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace TNL;
   using namespace TNL::Containers;
   using namespace TNL::Algorithms;

   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< TriangleMesh >;
   using GlobalIndexType = typename TriangleMesh::GlobalIndexType;
   using LocalIndexType = typename TriangleMesh::LocalIndexType;
   using PointType = typename TriangleMesh::PointType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, DecomposerVersion >;
   
   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< TriangleMesh::getMeshDimension() >();

   // Find the number of output points and cells as well as
   // starting indeces at which every cell will start writing new decomposed points and cells
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indeces( inCellsCount );
   auto setCounts = [&] ( GlobalIndexType i ) {
      const auto cell = inMesh.template getEntity< TriangleMesh::getMeshDimension() >( i );
      indeces[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
   };
   ParallelFor< Devices::Host >::exec( 0, inCellsCount, setCounts );
   const auto lastCounts = indeces[ indeces.getSize() - 1 ];
   auto reduction = [] ( const IndexPair& a, const IndexPair& b ) -> IndexPair {
      return { a.first + b.first, a.second + b.second };
   };
   inplaceExclusiveScan( indeces, 0, indeces.getSize(), reduction, std::make_pair( 0, 0 ) );
   const auto lastIndexPair = indeces[ indeces.getSize() - 1 ];
   const GlobalIndexType outPointsCount = inPointsCount + lastIndexPair.first + lastCounts.first;
   const GlobalIndexType outCellsCount = lastIndexPair.second + lastCounts.second;
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // Copy the points from inMesh to outMesh
   auto copyPoint = [&] ( GlobalIndexType i ) mutable {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   };
   ParallelFor< Devices::Host >::exec( 0, inPointsCount, copyPoint );

   // Decompose each cell
   auto decomposeCell = [&] ( GlobalIndexType i ) mutable {
      const auto cell = inMesh.template getEntity< TriangleMesh::getMeshDimension() >( i );
      const auto indexPair = indeces[ i ];

      // Lambda for adding new points
      GlobalIndexType setPointIndex = inPointsCount + indexPair.first;
      auto addPoint = [&] ( const PointType& point ) {
         const auto pointIdx = setPointIndex++;
         meshBuilder.setPoint( pointIdx, point );
         return pointIdx;
      };

      // Lambda for adding new cells
      GlobalIndexType setCellIndex = indexPair.second;
      auto addCell = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
         auto entitySeed = meshBuilder.getCellSeed( setCellIndex++ );
         entitySeed.setCornerId( 0, v0 );
         entitySeed.setCornerId( 1, v1 );
         entitySeed.setCornerId( 2, v2 );
      };

      EntityDecomposer::decompose( cell, addPoint, addCell );
   };
   ParallelFor< Devices::Host >::exec( 0, inCellsCount, decomposeCell );

   return meshBuilder;
}

template< EntityDecomposerVersion DecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto // returns Mesh
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   
   TriangleMesh outMesh;
   auto meshBuilder = decomposeMesh< DecomposerVersion >( inMesh );
   meshBuilder.build( outMesh );
   return outMesh;
}

// Polyhedral Mesh
template< typename ParentConfig >
struct TetrahedronConfig: public ParentConfig
{
   using CellTopology = Topologies::Tetrahedron;
};

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubdecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto // returns MeshBuilder
decomposeMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using namespace TNL;
   using namespace TNL::Containers;
   using namespace TNL::Algorithms;

   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< TetrahedronMesh >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;
   using PointType = typename TetrahedronMesh::PointType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polyhedron, DecomposerVersion, SubdecomposerVersion >;
   
   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< TetrahedronMesh::getMeshDimension() >();

   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indeces( inCellsCount );

   // Find the number of output points and cells as well as
   // starting indeces at which every cell will start writing new decomposed points and cells
   auto setCounts = [&] ( GlobalIndexType i ) {
      const auto cell = inMesh.template getEntity< TetrahedronMesh::getMeshDimension() >( i );
      indeces[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
   };
   ParallelFor< Devices::Host >::exec( 0, inCellsCount, setCounts );
   const auto lastCounts = indeces[ indeces.getSize() - 1 ];
   auto reduction = [] ( const IndexPair& a, const IndexPair& b ) -> IndexPair {
      return { a.first + b.first, a.second + b.second };
   };
   inplaceExclusiveScan( indeces, 0, indeces.getSize(), reduction, std::make_pair( 0, 0 ) );
   const auto lastIndexPair = indeces[ indeces.getSize() - 1 ];
   const GlobalIndexType outPointsCount = inPointsCount + lastIndexPair.first + lastCounts.first;
   const GlobalIndexType outCellsCount = lastIndexPair.second + lastCounts.second;
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // Copy the points from inMesh to outMesh
   auto copyPoint = [&] ( GlobalIndexType i ) mutable {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   };
   ParallelFor< Devices::Host >::exec( 0, inPointsCount, copyPoint );

   // Decompose each cell
   auto decomposeCell = [&] ( GlobalIndexType i ) mutable {
      const auto cell = inMesh.template getEntity< TetrahedronMesh::getMeshDimension() >( i );
      const auto indexPair = indeces[ i ];

      // Lambda for adding new points
      GlobalIndexType setPointIndex = inPointsCount + indexPair.first;
      auto addPoint = [&] ( const PointType& point ) {
         const auto pointIdx = setPointIndex++;
         meshBuilder.setPoint( pointIdx, point );
         return pointIdx;
      };

      // Lambda for adding new cells
      GlobalIndexType setCellIndex = indexPair.second;
      auto addCell = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2, GlobalIndexType v3 ) {
         auto entitySeed = meshBuilder.getCellSeed( setCellIndex++ );
         entitySeed.setCornerId( 0, v0 );
         entitySeed.setCornerId( 1, v1 );
         entitySeed.setCornerId( 2, v2 );
         entitySeed.setCornerId( 3, v3 );
      };

      EntityDecomposer::decompose( cell, addPoint, addCell );
   };
   ParallelFor< Devices::Host >::exec( 0, inCellsCount, decomposeCell );

   return meshBuilder;
}

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubDecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto // returns Mesh
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   
   TetrahedronMesh outMesh;
   auto meshBuilder = decomposeMesh< DecomposerVersion, SubDecomposerVersion >( inMesh );
   meshBuilder.build( outMesh );
   return outMesh;
}

} // namespace Meshes
} // namespace TNL