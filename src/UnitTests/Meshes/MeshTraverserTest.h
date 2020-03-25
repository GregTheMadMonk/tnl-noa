#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Traverser.h>

namespace MeshTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

// FIXME: Traverser does not work with Id = void
//class TestQuadrilateralMeshConfig : public DefaultConfig< Topologies::Quadrilateral >
class TestQuadrilateralMeshConfig : public DefaultConfig< Topologies::Quadrilateral, 2, double, int, int >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return ( SubentityDimensions % 2 != 0 ); }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
};

// FIXME: Traverser does not work with Id = void
//class TestHexahedronMeshConfig : public DefaultConfig< Topologies::Hexahedron >
class TestHexahedronMeshConfig : public DefaultConfig< Topologies::Hexahedron, 3, double, int, int >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) {  return ( SubentityDimensions % 2 != 0 ); }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
};

struct TestEntitiesProcessor
{
   template< typename Mesh, typename UserData, typename Entity >
   __cuda_callable__
   static void processEntity( const Mesh& mesh, UserData& userData, const Entity& entity )
   {
      userData[ entity.getIndex() ] += 1;
   }
};

template< typename EntityType, typename DeviceMeshPointer, typename HostArray >
void testCudaTraverser( const DeviceMeshPointer& deviceMeshPointer,
                        const HostArray& host_array_boundary,
                        const HostArray& host_array_interior,
                        const HostArray& host_array_all )
{
   using MeshType = typename DeviceMeshPointer::ObjectType;
   Traverser< MeshType, EntityType > traverser;

   Containers::Array< int, Devices::Cuda > array_boundary( deviceMeshPointer->template getEntitiesCount< EntityType >() );
   Containers::Array< int, Devices::Cuda > array_interior( deviceMeshPointer->template getEntitiesCount< EntityType >() );
   Containers::Array< int, Devices::Cuda > array_all     ( deviceMeshPointer->template getEntitiesCount< EntityType >() );

   array_boundary.setValue( 0 );
   array_interior.setValue( 0 );
   array_all     .setValue( 0 );

   traverser.template processBoundaryEntities< TestEntitiesProcessor >( deviceMeshPointer, array_boundary.getView() );
   traverser.template processInteriorEntities< TestEntitiesProcessor >( deviceMeshPointer, array_interior.getView() );
   traverser.template processAllEntities     < TestEntitiesProcessor >( deviceMeshPointer, array_all.getView() );

   EXPECT_EQ( array_boundary, host_array_boundary );
   EXPECT_EQ( array_interior, host_array_interior );
   EXPECT_EQ( array_all,      host_array_all      );
}

TEST( MeshTest, RegularMeshOfQuadrilateralsTest )
{
   using QuadrilateralMeshEntityType = MeshEntity< TestQuadrilateralMeshConfig, Devices::Host, Topologies::Quadrilateral >;
   using EdgeMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 3 ), ySize( 4 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   using TestQuadrilateralMesh = Mesh< TestQuadrilateralMeshConfig >;
   Pointers::SharedPointer< TestQuadrilateralMesh > meshPointer;
   MeshBuilder< TestQuadrilateralMesh > meshBuilder;
   meshBuilder.setPointsCount( numberOfVertices );
   meshBuilder.setCellsCount( numberOfCells );

   /****
    * Setup vertices
    */
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
      meshBuilder.setPoint( j * ( xSize + 1 ) + i, PointType( i * hx, j * hy ) );

   /****
    * Setup cells
    */
   IndexType cellIdx( 0 );
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType vertex0 = j * ( xSize + 1 ) + i;
      const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
      const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i + 1;
      const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i;

      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 2, vertex2 );
      meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 3, vertex3 );
   }

   ASSERT_TRUE( meshBuilder.build( *meshPointer ) );

   // traversers for all test cases
   Traverser< TestQuadrilateralMesh, QuadrilateralMeshEntityType > traverser_cells;
   Traverser< TestQuadrilateralMesh, EdgeMeshEntityType > traverser_edges;
   Traverser< TestQuadrilateralMesh, VertexMeshEntityType > traverser_vertices;

   // arrays for all test cases
   Containers::Array< int > array_cells_boundary( meshPointer->template getEntitiesCount< 2 >() );
   Containers::Array< int > array_cells_interior( meshPointer->template getEntitiesCount< 2 >() );
   Containers::Array< int > array_cells_all     ( meshPointer->template getEntitiesCount< 2 >() );

   Containers::Array< int > array_edges_boundary( meshPointer->template getEntitiesCount< 1 >() );
   Containers::Array< int > array_edges_interior( meshPointer->template getEntitiesCount< 1 >() );
   Containers::Array< int > array_edges_all     ( meshPointer->template getEntitiesCount< 1 >() );

   Containers::Array< int > array_vertices_boundary( meshPointer->template getEntitiesCount< 0 >() );
   Containers::Array< int > array_vertices_interior( meshPointer->template getEntitiesCount< 0 >() );
   Containers::Array< int > array_vertices_all     ( meshPointer->template getEntitiesCount< 0 >() );

   // reset all arrays
   array_cells_boundary.setValue( 0 );
   array_cells_interior.setValue( 0 );
   array_cells_all     .setValue( 0 );

   array_edges_boundary.setValue( 0 );
   array_edges_interior.setValue( 0 );
   array_edges_all     .setValue( 0 );

   array_vertices_boundary.setValue( 0 );
   array_vertices_interior.setValue( 0 );
   array_vertices_all     .setValue( 0 );

   // traverse for all test cases
   traverser_cells.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_cells_boundary.getView() );
   traverser_cells.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_cells_interior.getView() );
   traverser_cells.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_cells_all.getView() );

   traverser_edges.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_edges_boundary.getView() );
   traverser_edges.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_edges_interior.getView() );
   traverser_edges.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_edges_all.getView() );

   traverser_vertices.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_vertices_boundary.getView() );
   traverser_vertices.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_vertices_interior.getView() );
   traverser_vertices.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_vertices_all.getView() );

   // test traversing cells
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType idx = j * xSize + i;
      if( j == 0 || j == ySize - 1 || i == 0 || i == xSize - 1 ) {
         EXPECT_EQ( array_cells_boundary[ idx ], 1 );
         EXPECT_EQ( array_cells_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_cells_boundary[ idx ], 0 );
         EXPECT_EQ( array_cells_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_cells_all[ idx ], 1 );
   }

   // test traversing edges
   // (edges are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 1 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 1 >( idx ) ) {
         EXPECT_EQ( array_edges_boundary[ idx ], 1 );
         EXPECT_EQ( array_edges_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_edges_boundary[ idx ], 0 );
         EXPECT_EQ( array_edges_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_edges_all[ idx ], 1 );
   }

   // test traversing vertices
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
   {
      const IndexType idx = j * (xSize + 1) + i;
      if( j == 0 || j == ySize || i == 0 || i == xSize ) {
         EXPECT_EQ( array_vertices_boundary[ idx ], 1 );
         EXPECT_EQ( array_vertices_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_vertices_boundary[ idx ], 0 );
         EXPECT_EQ( array_vertices_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_vertices_all[ idx ], 1 );
   }

   // test traverser with CUDA
#ifdef HAVE_CUDA
   using DeviceMesh = Mesh< TestQuadrilateralMeshConfig, Devices::Cuda >;
   Pointers::SharedPointer< DeviceMesh > deviceMeshPointer;
   *deviceMeshPointer = *meshPointer;

   testCudaTraverser< QuadrilateralMeshEntityType >( deviceMeshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testCudaTraverser< EdgeMeshEntityType          >( deviceMeshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testCudaTraverser< VertexMeshEntityType        >( deviceMeshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );
#endif
}

TEST( MeshTest, RegularMeshOfHexahedronsTest )
{
   using HexahedronMeshEntityType = MeshEntity< TestHexahedronMeshConfig, Devices::Host, Topologies::Hexahedron >;
   using QuadrilateralMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 3 ), ySize( 4 ), zSize( 5 );
   const RealType width( 1.0 ), height( 1.0 ), depth( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize ),
                  hz( depth / ( RealType ) zSize );
   const IndexType numberOfCells = xSize * ySize * zSize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 ) * ( zSize + 1 );

   using TestHexahedronMesh = Mesh< TestHexahedronMeshConfig >;
   Pointers::SharedPointer< TestHexahedronMesh > meshPointer;
   MeshBuilder< TestHexahedronMesh > meshBuilder;
   meshBuilder.setPointsCount( numberOfVertices );
   meshBuilder.setCellsCount( numberOfCells );

   /****
    * Setup vertices
    */
   for( IndexType k = 0; k <= zSize; k++ )
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
      meshBuilder.setPoint( k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i, PointType( i * hx, j * hy, k * hz ) );

   /****
    * Setup cells
    */
   IndexType cellIdx( 0 );
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType vertex0 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
      const IndexType vertex1 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
      const IndexType vertex2 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
      const IndexType vertex3 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;
      const IndexType vertex4 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
      const IndexType vertex5 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
      const IndexType vertex6 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
      const IndexType vertex7 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;

      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 2, vertex2 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 3, vertex3 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 4, vertex4 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 5, vertex5 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 6, vertex6 );
      meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 7, vertex7 );
   }

   ASSERT_TRUE( meshBuilder.build( *meshPointer ) );

   // traversers for all test cases
   Traverser< TestHexahedronMesh, HexahedronMeshEntityType > traverser_cells;
   Traverser< TestHexahedronMesh, QuadrilateralMeshEntityType > traverser_faces;
   Traverser< TestHexahedronMesh, EdgeMeshEntityType > traverser_edges;
   Traverser< TestHexahedronMesh, VertexMeshEntityType > traverser_vertices;

   // arrays for all test cases
   Containers::Array< int > array_cells_boundary( meshPointer->template getEntitiesCount< 3 >() );
   Containers::Array< int > array_cells_interior( meshPointer->template getEntitiesCount< 3 >() );
   Containers::Array< int > array_cells_all     ( meshPointer->template getEntitiesCount< 3 >() );

   Containers::Array< int > array_faces_boundary( meshPointer->template getEntitiesCount< 2 >() );
   Containers::Array< int > array_faces_interior( meshPointer->template getEntitiesCount< 2 >() );
   Containers::Array< int > array_faces_all     ( meshPointer->template getEntitiesCount< 2 >() );

   Containers::Array< int > array_edges_boundary( meshPointer->template getEntitiesCount< 1 >() );
   Containers::Array< int > array_edges_interior( meshPointer->template getEntitiesCount< 1 >() );
   Containers::Array< int > array_edges_all     ( meshPointer->template getEntitiesCount< 1 >() );

   Containers::Array< int > array_vertices_boundary( meshPointer->template getEntitiesCount< 0 >() );
   Containers::Array< int > array_vertices_interior( meshPointer->template getEntitiesCount< 0 >() );
   Containers::Array< int > array_vertices_all     ( meshPointer->template getEntitiesCount< 0 >() );

   // reset all arrays
   array_cells_boundary.setValue( 0 );
   array_cells_interior.setValue( 0 );
   array_cells_all     .setValue( 0 );

   array_faces_boundary.setValue( 0 );
   array_faces_interior.setValue( 0 );
   array_faces_all     .setValue( 0 );

   array_edges_boundary.setValue( 0 );
   array_edges_interior.setValue( 0 );
   array_edges_all     .setValue( 0 );

   array_vertices_boundary.setValue( 0 );
   array_vertices_interior.setValue( 0 );
   array_vertices_all     .setValue( 0 );

   // traverse for all test cases
   traverser_cells.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_cells_boundary.getView() );
   traverser_cells.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_cells_interior.getView() );
   traverser_cells.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_cells_all.getView() );

   traverser_faces.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_faces_boundary.getView() );
   traverser_faces.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_faces_interior.getView() );
   traverser_faces.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_faces_all.getView() );

   traverser_edges.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_edges_boundary.getView() );
   traverser_edges.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_edges_interior.getView() );
   traverser_edges.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_edges_all.getView() );

   traverser_vertices.template processBoundaryEntities< TestEntitiesProcessor >( meshPointer, array_vertices_boundary.getView() );
   traverser_vertices.template processInteriorEntities< TestEntitiesProcessor >( meshPointer, array_vertices_interior.getView() );
   traverser_vertices.template processAllEntities     < TestEntitiesProcessor >( meshPointer, array_vertices_all.getView() );

   // test traversing cells
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType idx = k * xSize * ySize + j * xSize + i;
      if( k == 0 || k == zSize - 1 || j == 0 || j == ySize - 1 || i == 0 || i == xSize - 1 ) {
         EXPECT_EQ( array_cells_boundary[ idx ], 1 );
         EXPECT_EQ( array_cells_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_cells_boundary[ idx ], 0 );
         EXPECT_EQ( array_cells_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_cells_all[ idx ], 1 );
   }

   // test traversing faces
   // (faces are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 2 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 2 >( idx ) ) {
         EXPECT_EQ( array_faces_boundary[ idx ], 1 );
         EXPECT_EQ( array_faces_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_faces_boundary[ idx ], 0 );
         EXPECT_EQ( array_faces_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_faces_all[ idx ], 1 );
   }

   // test traversing edges
   // (edges are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 1 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 1 >( idx ) ) {
         EXPECT_EQ( array_edges_boundary[ idx ], 1 );
         EXPECT_EQ( array_edges_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_edges_boundary[ idx ], 0 );
         EXPECT_EQ( array_edges_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_edges_all[ idx ], 1 );
   }

   // test traversing vertices
   for( IndexType k = 0; k <= zSize; k++ )
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
   {
      const IndexType idx = k * (xSize + 1) * (ySize + 1) + j * (xSize + 1) + i;
      if( k == 0 || k == zSize || j == 0 || j == ySize || i == 0 || i == xSize ) {
         EXPECT_EQ( array_vertices_boundary[ idx ], 1 );
         EXPECT_EQ( array_vertices_interior[ idx ], 0 );
      }
      else {
         EXPECT_EQ( array_vertices_boundary[ idx ], 0 );
         EXPECT_EQ( array_vertices_interior[ idx ], 1 );
      }
      EXPECT_EQ( array_vertices_all[ idx ], 1 );
   }

   // test traverser with CUDA
#ifdef HAVE_CUDA
   using DeviceMesh = Mesh< TestHexahedronMeshConfig, Devices::Cuda >;
   Pointers::SharedPointer< DeviceMesh > deviceMeshPointer;
   *deviceMeshPointer = *meshPointer;

   testCudaTraverser< HexahedronMeshEntityType    >( deviceMeshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testCudaTraverser< QuadrilateralMeshEntityType >( deviceMeshPointer, array_faces_boundary, array_faces_interior, array_faces_all );
   testCudaTraverser< EdgeMeshEntityType          >( deviceMeshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testCudaTraverser< VertexMeshEntityType        >( deviceMeshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );
#endif
}

} // namespace MeshTest

#endif
