#pragma once

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>

namespace TNL {
namespace Meshes {


// Polygon Mesh
template< typename ParentConfig >
struct TriangleConfig: public ParentConfig
{
   using CellTopology = Topologies::Triangle;
};

enum class GetTriangleMeshVersion
{ 
   V1, V2 
};

/**
 * Triangles are made by connecting each edge to the centroid of cell.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto
getTriangleMesh_v1( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TriangleMesh::GlobalIndexType;
   using LocalIndexType = typename TriangleMesh::LocalIndexType;

   TriangleMesh outMesh;
   MeshBuilder< TriangleMesh > meshBuilder;
   
   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 2 >();

   // find the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount;
   GlobalIndexType outCellsCount = 0;

   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = cell.template getSubentitiesCount< 0 >();
      if( verticesCount == 3 ) { // cell is not decomposed as it's already a triangle
         outCellsCount++; // cell is just copied
      }
      else { // cell is decomposed as it has got more than 3 vertices
         outPointsCount++; // each decomposed cell has 1 new center point
         outCellsCount += verticesCount; // cell is decomposed into verticesCount number of triangles
      }
   }

   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // copy the points from inMesh to outMesh
   GlobalIndexType pointSetIdx = 0;
   for( ; pointSetIdx < inPointsCount; pointSetIdx++ ) {
      meshBuilder.setPoint( pointSetIdx, inMesh.getPoint( pointSetIdx ) );
   }

   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = cell.template getSubentitiesCount< 0 >();
      if( verticesCount == 3 ) { // cell is not decomposed as it's already a triangle
         // copy cell
         auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
         cellSeed.setCornerId( 0, cell.template getSubentityIndex< 0 >( 0 ) );
         cellSeed.setCornerId( 1, cell.template getSubentityIndex< 0 >( 1 ) );
         cellSeed.setCornerId( 2, cell.template getSubentityIndex< 0 >( 2 ) );
      }
      else { // cell is decomposed as it has got more than 3 vertices
         // add centroid of cell to outMesh
         const auto cellCenter = getEntityCenter( inMesh, cell );
         const auto cellCenterIdx = pointSetIdx++;
         meshBuilder.setPoint( cellCenterIdx, cellCenter );
         // decompose cell into triangles by connecting each edge to the centroid
         for( LocalIndexType j = 0, k = 1; k < verticesCount; j++, k++ ) {
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            cellSeed.setCornerId( 0, cell.template getSubentityIndex< 0 >( j ) );
            cellSeed.setCornerId( 1, cell.template getSubentityIndex< 0 >( k ) );
            cellSeed.setCornerId( 2, cellCenterIdx );
         }
         { // wrap around term
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            cellSeed.setCornerId( 0, cell.template getSubentityIndex< 0 >( verticesCount - 1 ) );
            cellSeed.setCornerId( 1, cell.template getSubentityIndex< 0 >( 0 ) );
            cellSeed.setCornerId( 2, cellCenterIdx );
         }
      }
   }
   
   meshBuilder.build( outMesh );
   return outMesh;
}

/**
 * Triangles are made by choosing the 0th point of cell and connecting each non-adjacent edge to it.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto
getTriangleMesh_v2( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TriangleMesh::GlobalIndexType;
   using LocalIndexType = typename TriangleMesh::LocalIndexType;

   TriangleMesh outMesh;
   MeshBuilder< TriangleMesh > meshBuilder;
   
   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 2 >();

   // outMesh keeps all the points of inMesh
   meshBuilder.setPointsCount( inPointsCount );
   for( GlobalIndexType i = 0; i < inPointsCount; i++ ) {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   }
   
   // find the number of cells in the outMesh
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = cell.template getSubentitiesCount< 0 >();
      outCellsCount += verticesCount - 2;
   }
   meshBuilder.setCellsCount( outCellsCount );

   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 2 >( i );
      const auto verticesCount = cell.template getSubentitiesCount< 0 >();
      const auto v0 = cell.template getSubentityIndex< 0 >( 0 );
      for( LocalIndexType j = 1, k = 2; k < verticesCount; j++, k++ ) {
         auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
         const auto v1 = cell.template getSubentityIndex< 0 >( j );
         const auto v2 = cell.template getSubentityIndex< 0 >( k );
         cellSeed.setCornerId( 0, v0 );
         cellSeed.setCornerId( 1, v1 );
         cellSeed.setCornerId( 2, v2 );
      }
   }
   
   meshBuilder.build( outMesh );
   return outMesh;
}

template< GetTriangleMeshVersion version,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   switch( version )
   {
      case GetTriangleMeshVersion::V1: return getTriangleMesh_v1( inMesh );
      case GetTriangleMeshVersion::V2: return getTriangleMesh_v2( inMesh );
   }
}

// Polyhedral Mesh
template< typename ParentConfig >
struct TetrahedronConfig: public ParentConfig
{
   using CellTopology = Topologies::Tetrahedron;
};

enum class GetTetrahedronMeshVersion
{ 
   V1, V2, V3, V4, V5 
};

/**
 * Tetrahedrons are made by connecting the triangles of faces to the centroid of cell.
 * Triangles of a face are made by connecting each edge to the centroid of a face.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getTetrahedronMesh_v1( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;

   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // count the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount + inCellsCount; // for each cell, there is a new point (centroid of cell)
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         if( verticesCount == 3 ) { // face is not decomposed as it's already a triangle
            outCellsCount++;
         }
         else { // face is decomposed as it has got more than 3 vertices
            outPointsCount++; // for each face of cell, there is a new point (centroid of face)
            outCellsCount += verticesCount; // there is verticesCount number of tetrahedrons per face
         }
      }
   }
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // copy the points from inMesh to outMesh
   GlobalIndexType pointSetIdx = 0;
   for( ; pointSetIdx < inPointsCount; pointSetIdx++ ) {
      meshBuilder.setPoint( pointSetIdx, inMesh.getPoint( pointSetIdx ) );
   }

   // set up cell seeds for the outMesh
   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );

      // centroid of the cell connects to triangles of its faces, making tetrahedrons
      const auto cellCenter = getEntityCenter( inMesh, cell );
      const auto cellCenterIdx = pointSetIdx++;
      meshBuilder.setPoint( cellCenterIdx, cellCenter );

      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();

         if( verticesCount == 3 ) { // face is a triangle
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            cellSeed.setCornerId( 0, face.template getSubentityIndex< 0 >( 0 ) );
            cellSeed.setCornerId( 1, face.template getSubentityIndex< 0 >( 1 ) );
            cellSeed.setCornerId( 2, face.template getSubentityIndex< 0 >( 2 ) );
            cellSeed.setCornerId( 3, cellCenterIdx );
         }
         else { // face is decomposed into triangles as it has got more than 3 vertices
            // centroid of the face connects to its edges, making triangles
            const auto faceCenter = getEntityCenter( inMesh, face );
            const auto faceCenterIdx = pointSetIdx++;
            meshBuilder.setPoint( faceCenterIdx, faceCenter );

            for( LocalIndexType k = 0, m = 1; m < verticesCount; k++, m++ ) {
               auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
               cellSeed.setCornerId( 0, face.template getSubentityIndex< 0 >( k ) );
               cellSeed.setCornerId( 1, face.template getSubentityIndex< 0 >( m ) );
               cellSeed.setCornerId( 2, faceCenterIdx );
               cellSeed.setCornerId( 3, cellCenterIdx );
            }
            { // wrap around term
               auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
               cellSeed.setCornerId( 0, face.template getSubentityIndex< 0 >( verticesCount - 1 ) );
               cellSeed.setCornerId( 1, face.template getSubentityIndex< 0 >( 0 ) );
               cellSeed.setCornerId( 2, faceCenterIdx );
               cellSeed.setCornerId( 3, cellCenterIdx );
            }
         }
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

/**
 * Tetrahedrons are made by connecting the triangles of faces to the centroid of cell.
 * Triangles of a face are made by choosing the 0th point of a face and connecting each of its non-adjacent edges to it.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getTetrahedronMesh_v2( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;

   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // count the number of points and cells in the outMesh
   GlobalIndexType outPointsCount = inPointsCount + inCellsCount; // for each cell, there is a new point (centroid of cell)
   GlobalIndexType outCellsCount = 0;

   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();

      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         outCellsCount += verticesCount - 2; // there is verticesCount - 2 number of tetrahedrons per face
      }
   }
   meshBuilder.setPointsCount( outPointsCount );
   meshBuilder.setCellsCount( outCellsCount );

   // copy the points from inMesh to outMesh
   GlobalIndexType pointSetIdx = 0;
   for( ; pointSetIdx < inPointsCount; pointSetIdx++ ) {
      meshBuilder.setPoint( pointSetIdx, inMesh.getPoint( pointSetIdx ) );
   }

   // set up cell seeds for the outMesh
   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );

      // centroid of the cell connects to triangles of its faces, making tetrahedrons
      const auto cellCenter = getEntityCenter( inMesh, cell );
      const auto cellCenterIdx = pointSetIdx++;
      meshBuilder.setPoint( cellCenterIdx, cellCenter );

      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0 = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         for( LocalIndexType k = 1, m = 2; m < verticesCount; k++, m++ ) {
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            const auto v1 = face.template getSubentityIndex< 0 >( k );
            const auto v2 = face.template getSubentityIndex< 0 >( m );
            cellSeed.setCornerId( 0, v0 );
            cellSeed.setCornerId( 1, v1 );
            cellSeed.setCornerId( 2, v2 );
            cellSeed.setCornerId( 3, cellCenterIdx );
         }
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

/**
 * Tetrahedrons are made by choosing the 0th point of a cell and connecting each triangle of its faces to it.
 * Triangles of a face are made by choosing the 0th point of a face and connecting each of its non-adjacent edges to it.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getTetrahedronMesh_v3( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;

   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // outMesh keeps all the points of inMesh
   meshBuilder.setPointsCount( inPointsCount );
   for( GlobalIndexType i = 0; i < inPointsCount; i++ ) {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   }

   // count the number of cells in the outMesh
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         outCellsCount += verticesCount - 2; // there is verticesCount - 2 number of tetrahedrons per face
      }
   }
   meshBuilder.setCellsCount( outCellsCount );

   // set up cell seeds for the outMesh
   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto v3 = cell.template getSubentityIndex< 0 >( 0 ); // point of the cell connects to triangles of its faces, making tetrahedrons
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0 = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         for( LocalIndexType k = 1, m = 2; m < verticesCount; k++, m++ ) {
            const auto v1 = face.template getSubentityIndex< 0 >( k ); 
            const auto v2 = face.template getSubentityIndex< 0 >( m );
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            cellSeed.setCornerId( 0, v0 );
            cellSeed.setCornerId( 1, v1 );
            cellSeed.setCornerId( 2, v2 );
            cellSeed.setCornerId( 3, v3 );
         }
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

/**
 * Tetrahedrons are made by choosing the 0th point of a cell and connecting each triangle of its faces to it.
 * Triangles of a face are made by choosing the 0th point of a face and connecting each of its non-adjacent edges to it.
 * Additionally, for every tetrahedron, it is checked whether it has non-zero volume. 
 * If not, given tetrahedron is redundant, thus left out.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getTetrahedronMesh_v4( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;
   using RealType = typename TetrahedronMesh::RealType;
   constexpr RealType precision{ 1e-6 };

   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // outMesh keeps all the points of inMesh
   meshBuilder.setPointsCount( inPointsCount );
   for( GlobalIndexType i = 0; i < inPointsCount; i++ ) {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   }

   // count the number of cells in the outMesh
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto v3Idx = cell.template getSubentityIndex< 0 >( 0 ); // point of the cell connects to triangles of its faces, making tetrahedrons
      const auto& v3 = inMesh.getPoint( v3Idx );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0Idx = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         const auto& v0 = inMesh.getPoint( v0Idx );
         for( LocalIndexType k = 1, m = 2; m < verticesCount; k++, m++ ) {
            const auto v1Idx = face.template getSubentityIndex< 0 >( k );
            const auto& v1 = inMesh.getPoint( v1Idx ); 
            const auto v2Idx = face.template getSubentityIndex< 0 >( m );
            const auto& v2 = inMesh.getPoint( v2Idx );
            const auto volume = getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
            if( volume > precision ) { // Leave out redundant tetrahedrons with zero volume
               outCellsCount++;
            }
         }
      }
   }
   meshBuilder.setCellsCount( outCellsCount );

   // set up cell seeds for the outMesh
   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto v3Idx = cell.template getSubentityIndex< 0 >( 0 ); // point of the cell connects to triangles of its faces, making tetrahedrons
      const auto& v3 = inMesh.getPoint( v3Idx );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0Idx = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         const auto& v0 = inMesh.getPoint( v0Idx );
         for( LocalIndexType k = 1, m = 2; m < verticesCount; k++, m++ ) {
            const auto v1Idx = face.template getSubentityIndex< 0 >( k );
            const auto& v1 = inMesh.getPoint( v1Idx ); 
            const auto v2Idx = face.template getSubentityIndex< 0 >( m );
            const auto& v2 = inMesh.getPoint( v2Idx );
            const auto volume = getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
            if( volume > precision ) { // Leave out redundant tetrahedrons with zero volume
               auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
               cellSeed.setCornerId( 0, v0Idx );
               cellSeed.setCornerId( 1, v1Idx );
               cellSeed.setCornerId( 2, v2Idx );
               cellSeed.setCornerId( 3, v3Idx );
            }
         }
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

/**
 * Tetrahedrons are made by choosing the 0th point of a cell and connecting each triangle of its faces to it.
 * Triangles of a face are made by choosing the 0th point of a face and connecting each of its non-adjacent edges to it.
 * Additionally, for the first tetrahedron of a face, it is checked whether it has non-zero volume. 
 * If not, all the tetrahedrons of given face are redundant, thus left out.
 * It is also assumed that all of the faces are planar.
 */
template< typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getTetrahedronMesh_v5( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using LocalIndexType = typename TetrahedronMesh::LocalIndexType;
   using RealType = typename TetrahedronMesh::RealType;
   constexpr RealType precision{ 1e-6 };

   TetrahedronMesh outMesh;
   MeshBuilder< TetrahedronMesh > meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< 3 >();

   // outMesh keeps all the points of inMesh
   meshBuilder.setPointsCount( inPointsCount );
   for( GlobalIndexType i = 0; i < inPointsCount; i++ ) {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   }

   // count the number of cells in the outMesh
   GlobalIndexType outCellsCount = 0;
   for( GlobalIndexType i = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto v3Idx = cell.template getSubentityIndex< 0 >( 0 ); // point of the cell connects to triangles of its faces, making tetrahedrons
      const auto& v3 = inMesh.getPoint( v3Idx );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0Idx = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         const auto& v0 = inMesh.getPoint( v0Idx );
         const auto& v1 = inMesh.getPoint( face.template getSubentityIndex< 0 >( 1 ) );
         const auto& v2 = inMesh.getPoint( face.template getSubentityIndex< 0 >( 2 ) );
         const auto volume = getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
         if( volume > precision ) { // Leave out redundant faces with zero volume tetrahedrons (It is expected that faces are planar)
            outCellsCount += verticesCount - 2; // there is verticesCount - 2 number of tetrahedrons per face
         } 
      }
   }
   meshBuilder.setCellsCount( outCellsCount );

   // set up cell seeds for the outMesh
   for( GlobalIndexType i = 0, cellSeedIdx = 0; i < inCellsCount; i++ ) {
      const auto cell = inMesh.template getEntity< 3 >( i );
      const auto v3Idx = cell.template getSubentityIndex< 0 >( 0 ); // point of the cell connects to triangles of its faces, making tetrahedrons
      const auto& v3 = inMesh.getPoint( v3Idx );
      const auto facesCount = cell.template getSubentitiesCount< 2 >();
      for( LocalIndexType j = 0; j < facesCount; j++ ) {
         const auto faceIdx = cell.template getSubentityIndex< 2 >( j );
         const auto face = inMesh.template getEntity< 2 >( faceIdx );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         const auto v0Idx = face.template getSubentityIndex< 0 >( 0 ); // point of the face connects to its edges, making triangles
         const auto& v0 = inMesh.getPoint( v0Idx );
         const auto& v1 = inMesh.getPoint( face.template getSubentityIndex< 0 >( 1 ) );
         const auto& v2 = inMesh.getPoint( face.template getSubentityIndex< 0 >( 2 ) );
         const auto volume = getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
         if( volume <= precision ) // Leave out redundant faces with zero volume tetrahedrons (It is expected that faces are planar)
            continue;
         for( LocalIndexType k = 1, m = 2; m < verticesCount; k++, m++ ) {
            const auto v1Idx = face.template getSubentityIndex< 0 >( k );
            const auto v2Idx = face.template getSubentityIndex< 0 >( m );
            auto & cellSeed = meshBuilder.getCellSeed( cellSeedIdx++ );
            cellSeed.setCornerId( 0, v0Idx );
            cellSeed.setCornerId( 1, v1Idx );
            cellSeed.setCornerId( 2, v2Idx );
            cellSeed.setCornerId( 3, v3Idx );
         }
      }
   }

   meshBuilder.build( outMesh );
   return outMesh;
}

template< GetTetrahedronMeshVersion version,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host > & inMesh )
{
   switch( version )
   {
      case GetTetrahedronMeshVersion::V1: return getTetrahedronMesh_v1( inMesh );
      case GetTetrahedronMeshVersion::V2: return getTetrahedronMesh_v2( inMesh );
      case GetTetrahedronMeshVersion::V3: return getTetrahedronMesh_v3( inMesh );
      case GetTetrahedronMeshVersion::V4: return getTetrahedronMesh_v4( inMesh );
      case GetTetrahedronMeshVersion::V5: return getTetrahedronMesh_v5( inMesh );
   }
}

} // namespace Meshes
} // namespace TNL