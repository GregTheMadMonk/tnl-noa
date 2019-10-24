#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <sstream>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Meshes/MeshBuilder.h>

namespace MeshTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

static const char* TEST_FILE_NAME = "test_MeshTest.tnl";

class TestTriangleMeshConfig : public DefaultConfig< Topologies::Triangle >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   //template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
};

class TestQuadrilateralMeshConfig : public DefaultConfig< Topologies::Quadrilateral >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return ( SubentityDimensions % 2 != 0 ); }
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; }
};

class TestTetrahedronMeshConfig : public DefaultConfig< Topologies::Tetrahedron >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; }
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) {  return ( SubentityDimensions % 2 != 0 ); }
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

template< typename Object1, typename Object2 >
void compareStringRepresentation( const Object1& obj1, const Object2& obj2 )
{
   std::stringstream str1, str2;
   str1 << obj1;
   str2 << obj2;
   EXPECT_EQ( str1.str(), str2.str() );
}

template< typename Object >
void testCopyAssignment( const Object& obj )
{
   static_assert( std::is_copy_constructible< Object >::value, "" );
   static_assert( std::is_copy_assignable< Object >::value, "" );

   Object new_obj_1( obj );
   EXPECT_EQ( new_obj_1, obj );
   Object new_obj_2;
   new_obj_2 = obj;
   EXPECT_EQ( new_obj_2, obj );
}

template< typename Mesh >
void testMeshOnCuda( const Mesh& mesh )
{
#ifdef HAVE_CUDA
   using DeviceMesh = Meshes::Mesh< typename Mesh::Config, Devices::Cuda >;

   // test host->CUDA copy
   DeviceMesh dmesh1( mesh );
   EXPECT_EQ( dmesh1, mesh );
   DeviceMesh dmesh2;
   dmesh2 = mesh;
   EXPECT_EQ( dmesh2, mesh );

   // test CUDA->CUDA copy
   testCopyAssignment( dmesh1 );

   // copy CUDA->host copy
   Mesh mesh2( dmesh1 );
   EXPECT_EQ( mesh2, mesh );
   Mesh mesh3;
   mesh3 = dmesh1;
   EXPECT_EQ( mesh2, mesh );

   // test load from file to CUDA
   ASSERT_NO_THROW( mesh.save( TEST_FILE_NAME ) );
   ASSERT_NO_THROW( dmesh1.load( TEST_FILE_NAME ) );
   EXPECT_EQ( dmesh1, mesh );

   // test save into file from CUDA
   ASSERT_NO_THROW( dmesh1.save( TEST_FILE_NAME ) );
   ASSERT_NO_THROW( mesh2.load( TEST_FILE_NAME ) );
   EXPECT_EQ( mesh2, mesh );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
#endif
}

template< typename Mesh >
void testEntities( const Mesh& mesh )
{
   using IndexType = typename Mesh::GlobalIndexType;

   // test that superentity accessors have been correctly bound
   for( IndexType i = 0; i < mesh.template getEntitiesCount< 0 >(); i++ ) {
      auto v1 = mesh.template getEntity< 0 >( i );
      auto& v2 = mesh.template getEntity< 0 >( i );
      EXPECT_EQ( v1, v2 );
      EXPECT_EQ( v1.template getSuperentitiesCount< Mesh::getMeshDimension() >(),
                 v2.template getSuperentitiesCount< Mesh::getMeshDimension() >() );
      for( IndexType s = 0; s < v1.template getSuperentitiesCount< Mesh::getMeshDimension() >(); s++ )
         EXPECT_EQ( v1.template getSuperentityIndex< Mesh::getMeshDimension() >( s ),
                    v2.template getSuperentityIndex< Mesh::getMeshDimension() >( s ) );
   }

   // test that subentity accessors have been correctly bound
   for( IndexType i = 0; i < mesh.template getEntitiesCount< Mesh::getMeshDimension() >(); i++ ) {
      auto c1 = mesh.template getEntity< Mesh::getMeshDimension() >( i );
      auto& c2 = mesh.template getEntity< Mesh::getMeshDimension() >( i );
      EXPECT_EQ( c1, c2 );
      EXPECT_EQ( c1.template getSubentitiesCount< 0 >(),
                 c2.template getSubentitiesCount< 0 >() );
      for( IndexType s = 0; s < c1.template getSubentitiesCount< 0 >(); s++ )
         EXPECT_EQ( c1.template getSubentityIndex< 0 >( s ),
                    c2.template getSubentityIndex< 0 >( s ) );
   }
}

template< typename Mesh >
void testFinishedMesh( const Mesh& mesh )
{
   Mesh mesh2;
   ASSERT_NO_THROW( mesh.save( TEST_FILE_NAME ) );
   ASSERT_NO_THROW( mesh2.load( TEST_FILE_NAME ) );
   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
   ASSERT_EQ( mesh, mesh2 );
   compareStringRepresentation( mesh, mesh2 );
   testCopyAssignment( mesh );
   testMeshOnCuda( mesh );
   testEntities( mesh );
}

TEST( MeshTest, TwoTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, Devices::Host, Topologies::Triangle >;
   using EdgeMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   /****
    * We set-up the following situation
            point2   edge3       point3
               |\-------------------|
               | \                  |
               |  \   triangle1     |
               |   \                |

                      ....
            edge1     edge0        edge4
                      ....


               |   triangle0     \  |
               |                  \ |
               ---------------------|
            point0   edge2        point1
    */

   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 ),
             point3( 1.0, 1.0 );

   typedef Mesh< TestTriangleMeshConfig > TriangleTestMesh;
   TriangleTestMesh mesh;
   MeshBuilder< TriangleTestMesh > meshBuilder;
   meshBuilder.setPointsCount( 4 );
   meshBuilder.setPoint( 0, point0 );
   meshBuilder.setPoint( 1, point1 );
   meshBuilder.setPoint( 2, point2 );
   meshBuilder.setPoint( 3, point3 );

   meshBuilder.setCellsCount( 2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 3 );
   ASSERT_TRUE( meshBuilder.build( mesh ) );

   EXPECT_EQ( mesh.getEntitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(),  5 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(),  4 );

   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).getPoint(),  point0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).getPoint(),  point1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).getPoint(),  point2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).getPoint(),  point3 );

   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).getVertexIndex( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).getVertexIndex( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).getVertexIndex( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).getVertexIndex( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).getVertexIndex( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).getVertexIndex( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).getVertexIndex( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).getVertexIndex( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).getVertexIndex( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).getVertexIndex( 1 ),  1 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 2 ),  0 );

   /*
    * Tests for the superentities layer.
    */
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 0 ),    1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 1 ),    2 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 1 ),    2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 2 ),    4 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 1 ),    1 );


   testFinishedMesh( mesh );
};

TEST( MeshTest, TetrahedronsTest )
{
   using TetrahedronMeshEntityType = MeshEntity< TestTetrahedronMeshConfig, Devices::Host, Topologies::Tetrahedron >;
   using TriangleMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   typedef Mesh< TestTetrahedronMeshConfig > TestTetrahedronMesh;
   TestTetrahedronMesh mesh;
   MeshBuilder< TestTetrahedronMesh > meshBuilder;
   meshBuilder.setPointsCount( 13 );
   meshBuilder.setPoint(  0, PointType(  0.000000, 0.000000, 0.000000 ) );
   meshBuilder.setPoint(  1, PointType(  0.000000, 0.000000, 8.000000 ) );
   meshBuilder.setPoint(  2, PointType(  0.000000, 8.000000, 0.000000 ) );
   meshBuilder.setPoint(  3, PointType( 15.000000, 0.000000, 0.000000 ) );
   meshBuilder.setPoint(  4, PointType(  0.000000, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  5, PointType( 15.000000, 0.000000, 8.000000 ) );
   meshBuilder.setPoint(  6, PointType( 15.000000, 8.000000, 0.000000 ) );
   meshBuilder.setPoint(  7, PointType( 15.000000, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  8, PointType(  7.470740, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  9, PointType(  7.470740, 0.000000, 8.000000 ) );
   meshBuilder.setPoint( 10, PointType(  7.504125, 8.000000, 0.000000 ) );
   meshBuilder.setPoint( 11, PointType(  7.212720, 0.000000, 0.000000 ) );
   meshBuilder.setPoint( 12, PointType( 11.184629, 3.987667, 3.985835 ) );

   /****
    * Setup the following tetrahedrons:
    * ( Generated by Netgen )
    *
    *  12        8        7        5
    *  12        7        8       10
    *  12       11        8        9
    *  10       11        2        8
    *  12        7        6        5
    *   9       12        5        8
    *  12       11        9        3
    *   9        4       11        8
    *  12        9        5        3
    *   1        2        0       11
    *   8       11        2        4
    *   1        2       11        4
    *   9        4        1       11
    *  10       11        8       12
    *  12        6        7       10
    *  10       11       12        3
    *  12        6        3        5
    *  12        3        6       10
    */

   meshBuilder.setCellsCount( 18 );
    //  12        8        7        5
   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 8 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 7 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 5 );

    //  12        7        8       10
   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 7 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3, 10 );

    //  12       11        8        9
   meshBuilder.getCellSeed( 2 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 3, 9 );

    //  10       11        2        8
   meshBuilder.getCellSeed( 3 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 3, 8 );

    //  12        7        6        5
   meshBuilder.getCellSeed( 4 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 1, 7 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 2, 6 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 3, 5 );

    //   9       12        5        8
   meshBuilder.getCellSeed( 5 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 1, 12 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 2, 5 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 3, 8 );

    //  12       11        9        3
   meshBuilder.getCellSeed( 6 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 2, 9 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 3, 3 );

    //   9        4       11        8
   meshBuilder.getCellSeed( 7 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 1, 4 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 2, 11 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 3, 8 );

    //  12        9        5        3
   meshBuilder.getCellSeed( 8 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 1, 9 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 2, 5 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 3, 3 );

    //   1        2        0       11
   meshBuilder.getCellSeed( 9 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 2, 0 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 3, 11 );

    //   8       11        2        4
   meshBuilder.getCellSeed( 10 ).setCornerId( 0, 8 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 3, 4 );

    //   1        2       11        4
   meshBuilder.getCellSeed( 11 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 2, 11 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 3, 4 );

    //   9        4        1       11
   meshBuilder.getCellSeed( 12 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 1, 4 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 2, 1 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 3, 11 );

    //  10       11        8       12
   meshBuilder.getCellSeed( 13 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 3, 12 );

    //  12        6        7       10
   meshBuilder.getCellSeed( 14 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 1, 6 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 2, 7 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 3, 10 );

    //  10       11       12        3
   meshBuilder.getCellSeed( 15 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 2, 12 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 3, 3 );

    //  12        6        3        5
   meshBuilder.getCellSeed( 16 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 1, 6 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 2, 3 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 3, 5 );

    //  12        3        6       10
   meshBuilder.getCellSeed( 17 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 1, 3 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 2, 6 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 3, 10 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   testFinishedMesh( mesh );
}

TEST( MeshTest, RegularMeshOfTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, Devices::Host, Topologies::Triangle >;
   using EdgeMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 5 ), ySize( 5 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = 2 * xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   typedef Mesh< TestTriangleMeshConfig > TestTriangleMesh;
   Mesh< TestTriangleMeshConfig > mesh;
   MeshBuilder< TestTriangleMesh > meshBuilder;
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
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i + 1;

         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
         meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 2, vertex2 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex1 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex2 );
         meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 2, vertex3 );
      }

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // Test cells -> vertices subentities
   cellIdx = 0;
   for( IndexType j = 0; j < ySize; j++ )
      for( IndexType i = 0; i < xSize; i++ )
      {
         const IndexType vertex0 = j * ( xSize + 1 ) + i;
         const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i + 1;

         const TriangleMeshEntityType& leftCell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 0 ), vertex0 );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 1 ), vertex1 );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 2 ), vertex2 );

         const TriangleMeshEntityType& rightCell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 0 ), vertex1 );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 1 ), vertex2 );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 2 ), vertex3 );
      }

   // Test vertices -> cells superentities
   for( IndexType j = 0; j <= ySize; j++ )
      for( IndexType i = 0; i <= xSize; i++ )
      {
         const IndexType vertexIndex = j * ( xSize + 1 ) + i;
         const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

         if( ( i == 0 && j == 0 ) || ( i == xSize && j == ySize ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 1 );
         }
         else if( ( i == 0 && j == ySize ) || ( i == xSize && j == 0 ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 2 );
         }
         else if( i == 0 || i == xSize || j == 0 || j == ySize ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 3 );
         }
         else {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 6 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 6 );
         }
      }

   testFinishedMesh( mesh );
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

   typedef Mesh< TestQuadrilateralMeshConfig > TestQuadrilateralMesh;
   TestQuadrilateralMesh mesh;
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

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // Test cells -> vertices subentities
   cellIdx = 0;
   for( IndexType j = 0; j < ySize; j++ )
      for( IndexType i = 0; i < xSize; i++ )
      {
         const IndexType vertex0 = j * ( xSize + 1 ) + i;
         const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i + 1;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i;

         const QuadrilateralMeshEntityType& cell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 0 ), vertex0 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 1 ), vertex1 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 2 ), vertex2 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 3 ), vertex3 );
      }

   // Test vertices -> cells superentities
   for( IndexType j = 0; j <= ySize; j++ )
      for( IndexType i = 0; i <= xSize; i++ )
      {
         const IndexType vertexIndex = j * ( xSize + 1 ) + i;
         const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

         if( ( i == 0 || i == xSize ) && ( j == 0 || j == ySize ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
         }
         else if( i == 0 || i == xSize || j == 0 || j == ySize ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - ( j == ySize || i == 0 || i == xSize ) ) * xSize + i - ( i == xSize ) - ( j == 0 || j == ySize ) );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 1 ),   ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
         }
         else {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - 1 ) * xSize + i - 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 1 ),   ( j - 1 ) * xSize + i     );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 2 ),   ( j     ) * xSize + i - 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 3 ),   ( j     ) * xSize + i     );
         }
      }

   testFinishedMesh( mesh );
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

   typedef Mesh< TestHexahedronMeshConfig > TestHexahedronMesh;
   TestHexahedronMesh mesh;
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

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // Test cells -> vertices subentities
   cellIdx = 0;
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

            const HexahedronMeshEntityType& cell = mesh.template getEntity< 3 >( cellIdx++ );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 0 ), vertex0 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 1 ), vertex1 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 2 ), vertex2 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 3 ), vertex3 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 4 ), vertex4 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 5 ), vertex5 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 6 ), vertex6 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 7 ), vertex7 );
         }

   // Test vertices -> cells superentities
   for( IndexType k = 0; k < zSize; k++ )
      for( IndexType j = 0; j <= ySize; j++ )
         for( IndexType i = 0; i <= xSize; i++ )
         {
            const IndexType vertexIndex = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

            if( ( i == 0 || i == xSize ) && ( j == 0 || j == ySize ) && ( k == 0 || k == zSize ) ) {
               EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 3 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
            }
            else if( i == 0 || i == xSize || j == 0 || j == ySize || k == 0 || k == zSize ) {
               if( ( i != 0 && i != xSize && j != 0 && j != ySize ) ||
                   ( i != 0 && i != xSize && k != 0 && k != zSize ) ||
                   ( j != 0 && j != ySize && k != 0 && k != zSize ) )
               {
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 5 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 8 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 4 );
                  if( k == 0 || k == zSize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i     );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i     );
                  }
                  else if( j == 0 || j == ySize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                  }
                  else {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                  }
               }
               else {
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 5 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 2 );
                  if( k != 0 && k != zSize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
                  }
                  else if( j != 0 && j != ySize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                  }
                  else {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                  }
               }
            }
            else {
               EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 6 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 12 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 8 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 4 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 5 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 6 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 7 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i     );
            }
         }

   testFinishedMesh( mesh );
}

} // namespace MeshTest

#endif
