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

class TestQuadrilateralMeshConfig : public DefaultConfig< Topologies::Quadrilateral >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return ( SubentityDimensions % 2 != 0 ); }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
};

class TestHexahedronMeshConfig : public DefaultConfig< Topologies::Hexahedron >
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
void testTraverser( const DeviceMeshPointer& deviceMeshPointer,
                    const HostArray& host_array_boundary,
                    const HostArray& host_array_interior,
                    const HostArray& host_array_all )
{
   using MeshType = typename DeviceMeshPointer::ObjectType;
   using DeviceType = typename MeshType::DeviceType;
   static_assert( std::is_same< DeviceType, typename DeviceMeshPointer::DeviceType >::value, "devices must be the same" );
   Traverser< MeshType, EntityType > traverser;

   Containers::Array< int, DeviceType > array_boundary( deviceMeshPointer->template getEntitiesCount< EntityType >() );
   Containers::Array< int, DeviceType > array_interior( deviceMeshPointer->template getEntitiesCount< EntityType >() );
   Containers::Array< int, DeviceType > array_all     ( deviceMeshPointer->template getEntitiesCount< EntityType >() );

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

   // set expected values
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType idx = j * xSize + i;
      if( j == 0 || j == ySize - 1 || i == 0 || i == xSize - 1 ) {
         array_cells_boundary[ idx ] = 1;
         array_cells_interior[ idx ] = 0;
      }
      else {
         array_cells_boundary[ idx ] = 0;
         array_cells_interior[ idx ] = 1;
      }
      array_cells_all[ idx ] = 1;
   }
   // (edges are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 1 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 1 >( idx ) ) {
         array_edges_boundary[ idx ] = 1;
         array_edges_interior[ idx ] = 0;
      }
      else {
         array_edges_boundary[ idx ] = 0;
         array_edges_interior[ idx ] = 1;
      }
      array_edges_all[ idx ] = 1;
   }
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
   {
      const IndexType idx = j * (xSize + 1) + i;
      if( j == 0 || j == ySize || i == 0 || i == xSize ) {
         array_vertices_boundary[ idx ] = 1;
         array_vertices_interior[ idx ] = 0;
      }
      else {
         array_vertices_boundary[ idx ] = 0;
         array_vertices_interior[ idx ] = 1;
      }
      array_vertices_all[ idx ] = 1;
   }

   // test traverser with host
   testTraverser< QuadrilateralMeshEntityType >( meshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testTraverser< EdgeMeshEntityType          >( meshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testTraverser< VertexMeshEntityType        >( meshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );

   // test traverser with CUDA
#ifdef HAVE_CUDA
   using DeviceMesh = Mesh< TestQuadrilateralMeshConfig, Devices::Cuda >;
   Pointers::SharedPointer< DeviceMesh > deviceMeshPointer;
   *deviceMeshPointer = *meshPointer;

   testTraverser< QuadrilateralMeshEntityType >( deviceMeshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testTraverser< EdgeMeshEntityType          >( deviceMeshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testTraverser< VertexMeshEntityType        >( deviceMeshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );
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

   // set expected values
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType idx = k * xSize * ySize + j * xSize + i;
      if( k == 0 || k == zSize - 1 || j == 0 || j == ySize - 1 || i == 0 || i == xSize - 1 ) {
         array_cells_boundary[ idx ] = 1;
         array_cells_interior[ idx ] = 0;
      }
      else {
         array_cells_boundary[ idx ] = 0;
         array_cells_interior[ idx ] = 1;
      }
      array_cells_all[ idx ] = 1;
   }
   // (faces are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 2 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 2 >( idx ) ) {
         array_faces_boundary[ idx ] = 1;
         array_faces_interior[ idx ] = 0;
      }
      else {
         array_faces_boundary[ idx ] = 0;
         array_faces_interior[ idx ] = 1;
      }
      array_faces_all[ idx ] = 1;
   }
   // (edges are not numbered systematically, so we just compare with isBoundaryEntity)
   for( IndexType idx = 0; idx < meshPointer->template getEntitiesCount< 1 >(); idx++ )
   {
      if( meshPointer->template isBoundaryEntity< 1 >( idx ) ) {
         array_edges_boundary[ idx ] = 1;
         array_edges_interior[ idx ] = 0;
      }
      else {
         array_edges_boundary[ idx ] = 0;
         array_edges_interior[ idx ] = 1;
      }
      array_edges_all[ idx ] = 1;
   }
   for( IndexType k = 0; k <= zSize; k++ )
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
   {
      const IndexType idx = k * (xSize + 1) * (ySize + 1) + j * (xSize + 1) + i;
      if( k == 0 || k == zSize || j == 0 || j == ySize || i == 0 || i == xSize ) {
         array_vertices_boundary[ idx ] = 1;
         array_vertices_interior[ idx ] = 0;
      }
      else {
         array_vertices_boundary[ idx ] = 0;
         array_vertices_interior[ idx ] = 1;
      }
      array_vertices_all[ idx ] = 1;
   }

   // test traverser with host
   testTraverser< HexahedronMeshEntityType    >( meshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testTraverser< QuadrilateralMeshEntityType >( meshPointer, array_faces_boundary, array_faces_interior, array_faces_all );
   testTraverser< EdgeMeshEntityType          >( meshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testTraverser< VertexMeshEntityType        >( meshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );

   // test traverser with CUDA
#ifdef HAVE_CUDA
   using DeviceMesh = Mesh< TestHexahedronMeshConfig, Devices::Cuda >;
   Pointers::SharedPointer< DeviceMesh > deviceMeshPointer;
   *deviceMeshPointer = *meshPointer;

   testTraverser< HexahedronMeshEntityType    >( deviceMeshPointer, array_cells_boundary, array_cells_interior, array_cells_all );
   testTraverser< QuadrilateralMeshEntityType >( deviceMeshPointer, array_faces_boundary, array_faces_interior, array_faces_all );
   testTraverser< EdgeMeshEntityType          >( deviceMeshPointer, array_edges_boundary, array_edges_interior, array_edges_all );
   testTraverser< VertexMeshEntityType        >( deviceMeshPointer, array_vertices_boundary, array_vertices_interior, array_vertices_all );
#endif
}

} // namespace MeshTest

#endif
