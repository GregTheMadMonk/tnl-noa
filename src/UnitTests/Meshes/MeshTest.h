#pragma once

#ifdef HAVE_GTEST
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/Topologies/MeshEdgeTopology.h>
#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/Topologies/MeshQuadrilateralTopology.h>
#include <TNL/Meshes/Topologies/MeshTetrahedronTopology.h>
#include <TNL/Meshes/Topologies/MeshHexahedronTopology.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>
#include <TNL/Meshes/MeshBuilder.h>

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

class TestTriangleMeshConfig : public MeshConfigBase< MeshTriangleTopology >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; };
   //template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; };
};

class TestQuadrilateralMeshConfig : public MeshConfigBase< MeshQuadrilateralTopology >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return ( SubentityDimensions % 2 != 0 ); };
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; };
};

class TestTetrahedronMeshConfig : public MeshConfigBase< MeshTetrahedronTopology >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) {  return ( SubentityDimensions % 2 != 0 ); };
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; };
};

class TestHexahedronMeshConfig : public MeshConfigBase< MeshHexahedronTopology >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) {  return ( SubentityDimensions % 2 != 0 ); };
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; };
};

TEST( MeshTest, TwoTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, MeshTriangleTopology >;
   using EdgeMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

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
   TriangleTestMesh mesh, mesh2;
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

   EXPECT_EQ( mesh.getNumberOfEntities< 2 >(),  2 );
   EXPECT_EQ( mesh.getNumberOfEntities< 1 >(),  5 );
   EXPECT_EQ( mesh.getNumberOfEntities< 0 >(),  4 );

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
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getNumberOfSuperentities< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 0 ),    1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 1 ),    2 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getNumberOfSuperentities< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 1 ),    2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 2 ),    4 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getNumberOfSuperentities< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 0 ).template getNumberOfSuperentities< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 1 ),    1 );


   //ASSERT_TRUE( mesh.save( "mesh.tnl" ) );
   //ASSERT_TRUE( mesh2.load( "mesh.tnl" ) );
   //ASSERT_TRUE( mesh == mesh2 );

   //mesh.print(std::cout );
   //mesh2.print(std::cout );
};

TEST( MeshTest, TetrahedronsTest )
{
   using TetrahedronMeshEntityType = MeshEntity< TestTetrahedronMeshConfig, MeshTetrahedronTopology >;
   using TriangleMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 3, RealType >::getType() ) );

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

   /*ASSERT_TRUE( mesh.save( "mesh.tnl" ) );
   ASSERT_TRUE( mesh2.load( "mesh.tnl" ) );
   ASSERT_TRUE( mesh == mesh2 );*/
   //mesh.print(std::cout );
}

TEST( MeshTest, RegularMeshOfTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, MeshTriangleTopology >;
   using EdgeMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

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
   //ASSERT_TRUE( mesh.save( "mesh-test.tnl" ) );
   //ASSERT_TRUE( mesh2.load( "mesh-test.tnl" ) );
   //ASSERT_TRUE( mesh == mesh2 );
   //mesh.print(std::cout );
}

TEST( MeshTest, RegularMeshOfQuadrilateralsTest )
{
   using QuadrilateralMeshEntityType = MeshEntity< TestQuadrilateralMeshConfig, MeshQuadrilateralTopology >;
   using EdgeMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename QuadrilateralMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

   const IndexType xSize( 5 ), ySize( 5 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   typedef Mesh< TestQuadrilateralMeshConfig > TestQuadrilateralMesh;
   TestQuadrilateralMesh mesh, mesh2;
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

   //ASSERT_TRUE( mesh.save( "mesh-test.tnl" ) );
   //ASSERT_TRUE( mesh2.load( "mesh-test.tnl" ) );
   //ASSERT_TRUE( mesh == mesh2 );
   //mesh.print(std::cout );
}

TEST( MeshTest, RegularMeshOfHexahedronsTest )
{
   using HexahedronMeshEntityType = MeshEntity< TestHexahedronMeshConfig, MeshHexahedronTopology >;
   using QuadrilateralMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 3, RealType >::getType() ) );

   const IndexType xSize( 5 ), ySize( 5 ), zSize( 5 );
   const RealType width( 1.0 ), height( 1.0 ), depth( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize ),
                  hz( depth / ( RealType ) zSize );
   const IndexType numberOfCells = xSize * ySize * zSize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 ) * ( zSize + 1 );

   typedef Mesh< TestHexahedronMeshConfig > TestHexahedronMesh;
   TestHexahedronMesh mesh, mesh2;
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
            const IndexType vertex2 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;
            const IndexType vertex3 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
            const IndexType vertex4 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const IndexType vertex5 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
            const IndexType vertex6 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;
            const IndexType vertex7 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;

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

   //meshInitializer.initMesh( mesh );
   /*ASSERT_TRUE( mesh.save( "mesh-test.tnl" ) );
   ASSERT_TRUE( mesh2.load( "mesh-test.tnl" ) );
   ASSERT_TRUE( mesh == mesh2 );*/
   //mesh.print(std::cout );
}

#endif
