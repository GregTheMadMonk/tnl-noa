#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <array>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/MeshBuilder.h>

namespace MeshOrderingTest {

using namespace TNL;
using namespace TNL::Meshes;

class TestTriangleMeshConfig
   : public MeshConfigBase< MeshTriangleTopology, 2, double, int, short int, int >
{
public:
   static constexpr bool entityStorage( int dimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool subentityStorage( EntityTopology, int SubentityDimensions ) { return true; };
   //template< typename EntityTopology > static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimensions ) { return true; };
   template< typename EntityTopology > static constexpr bool superentityStorage( EntityTopology, int SuperentityDimensions ) { return true; };
};

template< typename Device >
bool buildTriangleMesh( Mesh< TestTriangleMeshConfig, Device >& mesh )
{
   using TriangleMesh = Mesh< TestTriangleMeshConfig, Device >;
   using TriangleMeshEntityType = typename TriangleMesh::template EntityType< 2 >;
   using EdgeMeshEntityType = typename TriangleMesh::template EntityType< 1 >;
   using VertexMeshEntityType = typename TriangleMesh::template EntityType< 0 >;

   static_assert( TriangleMeshEntityType::template SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::template SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::template SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::template SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::template SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::template SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, double > >::value, "" );

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

   MeshBuilder< TriangleMesh > meshBuilder;
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
   return meshBuilder.build( mesh );
}

template< typename PermutationVector >
void testMesh( const Mesh< TestTriangleMeshConfig, Devices::Host >& mesh,
               const PermutationVector& vertexPermutation,
               const PermutationVector& edgePermutation,
               const PermutationVector& cellPermutation )
{
   using MeshType = Mesh< TestTriangleMeshConfig, Devices::Host >;
   using PointType = typename MeshType::PointType;

   ASSERT_EQ( vertexPermutation.getSize(), 4 );
   ASSERT_EQ( edgePermutation.getSize(),   5 );
   ASSERT_EQ( cellPermutation.getSize(),   2 );

   EXPECT_EQ( mesh.getEntitiesCount< 0 >(),  4 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(),  5 );
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(),  2 );

   // test points
   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 ),
             point3( 1.0, 1.0 );

   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).getPoint(),  point0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).getPoint(),  point1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).getPoint(),  point2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).getPoint(),  point3 );


   // test getIndex
   for( int i = 0; i < 4; i++ )
      EXPECT_EQ( mesh.template getEntity< 0 >( i ).getIndex(), i );
   for( int i = 0; i < 5; i++ )
      EXPECT_EQ( mesh.template getEntity< 1 >( i ).getIndex(), i );
   for( int i = 0; i < 2; i++ )
      EXPECT_EQ( mesh.template getEntity< 2 >( i ).getIndex(), i );


   // test subentities
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).getVertexIndex( 0 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).getVertexIndex( 1 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).getVertexIndex( 0 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).getVertexIndex( 1 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).getVertexIndex( 0 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).getVertexIndex( 1 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).getVertexIndex( 0 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).getVertexIndex( 1 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).getVertexIndex( 0 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).getVertexIndex( 1 ),  vertexPermutation[ 1 ] );

   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 2 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 1 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 2 ),  edgePermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 2 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 0 ),  edgePermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 1 ),  edgePermutation[ 4 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 2 ),  edgePermutation[ 0 ] );


   // test superentities
   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 2 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 2 ),  edgePermutation[ 4 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 2 ),  edgePermutation[ 3 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 4 ] );


   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );


   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );


   // test boundary tags
   const std::vector< int > boundaryFaces = {1, 2, 3, 4};
   const std::vector< int > interiorFaces = {0};
   EXPECT_EQ( mesh.template getBoundaryEntitiesCount< 1 >(), boundaryFaces.size() );
   for( size_t i = 0; i < boundaryFaces.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 1 >( edgePermutation[ boundaryFaces[ i ] ] ) );
      // boundary indices are always sorted so we can't test this
//      EXPECT_EQ( mesh.template getBoundaryEntityIndex< 1 >( i ), edgePermutation[ boundaryFaces[ i ] ] );
   }
   // Test interior faces
   EXPECT_EQ( mesh.template getInteriorEntitiesCount< 1 >(), interiorFaces.size() );
   for( size_t i = 0; i < interiorFaces.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 1 >( edgePermutation[ interiorFaces[ i ] ] ) );
      // boundary indices are always sorted so we can't test this
//      EXPECT_EQ( mesh.template getInteriorEntityIndex< 1 >( i ), edgePermutation[ interiorFaces[ i ] ] );
   }
}

TEST( MeshOrderingTest, TwoTrianglesTest )
{
   using MeshHost = Mesh< TestTriangleMeshConfig, Devices::Host >;
   using MeshCuda = Mesh< TestTriangleMeshConfig, Devices::Cuda >;

   MeshHost mesh;
   ASSERT_TRUE( buildTriangleMesh( mesh ) );

   // hack due to TNL::Containers::Vector not supporting initilizer lists
   std::array< int, 4 > _vertexIdentity { { 0, 1, 2, 3 } };
   std::array< int, 5 > _edgeIdentity   { { 0, 1, 2, 3, 4 } };
   std::array< int, 2 > _cellIdentity   { { 0, 1 } };

   std::array< int, 4 > _vertexPermutation { { 3, 2, 0, 1 } };
   std::array< int, 5 > _edgePermutation   { { 2, 0, 4, 1, 3 } };
   std::array< int, 2 > _cellPermutation   { { 1, 0 } };

   std::array< int, 4 > _vertexInversePermutation { { 2, 3, 1, 0 } };
   std::array< int, 5 > _edgeInversePermutation   { { 1, 3, 0, 4, 2 } };
   std::array< int, 2 > _cellInversePermutation   { { 1, 0 } };

   using PermutationVector = typename MeshHost::IndexPermutationVector;
   const PermutationVector vertexIdentity ( &_vertexIdentity[0], 4 );
   const PermutationVector edgeIdentity   ( &_edgeIdentity[0],   5 );
   const PermutationVector cellIdentity   ( &_cellIdentity[0],   2 );

   const PermutationVector vertexPermutation ( &_vertexPermutation[0], 4 );
   const PermutationVector edgePermutation   ( &_edgePermutation[0],   5 );
   const PermutationVector cellPermutation   ( &_cellPermutation[0],   2 );

   const PermutationVector vertexInversePermutation ( &_vertexInversePermutation[0], 4 );
   const PermutationVector edgeInversePermutation   ( &_edgeInversePermutation[0],   5 );
   const PermutationVector cellInversePermutation   ( &_cellInversePermutation[0],   2 );

   ASSERT_TRUE( mesh.template reorderEntities< 0 >( vertexPermutation, vertexInversePermutation ) );
   testMesh( mesh, vertexInversePermutation, edgeIdentity, cellIdentity );

   ASSERT_TRUE( mesh.template reorderEntities< 2 >( cellPermutation, cellInversePermutation ) );
   testMesh( mesh, vertexInversePermutation, edgeIdentity, cellInversePermutation );

   ASSERT_TRUE( mesh.template reorderEntities< 1 >( edgePermutation, edgeInversePermutation ) );
   testMesh( mesh, vertexInversePermutation, edgeInversePermutation, cellInversePermutation );

#ifdef HAVE_CUDA
   MeshCuda meshCuda;
   MeshHost mesh2;
   ASSERT_TRUE( buildTriangleMesh( mesh2 ) );
   meshCuda = mesh2;
   using PermutationCuda = typename PermutationVector::CudaType;

   PermutationCuda vertexPermutationCuda;
   ASSERT_TRUE( vertexPermutationCuda.setLike( vertexPermutation ) );
   vertexPermutationCuda = vertexPermutation;
   PermutationCuda edgePermutationCuda;
   ASSERT_TRUE( edgePermutationCuda.setLike( edgePermutation ) );
   edgePermutationCuda = edgePermutation;
   PermutationCuda cellPermutationCuda;
   ASSERT_TRUE( cellPermutationCuda.setLike( cellPermutation ) );
   cellPermutationCuda = cellPermutation;

   PermutationCuda vertexInversePermutationCuda;
   ASSERT_TRUE( vertexInversePermutationCuda.setLike( vertexInversePermutation ) );
   vertexInversePermutationCuda = vertexInversePermutation;
   PermutationCuda edgeInversePermutationCuda;
   ASSERT_TRUE( edgeInversePermutationCuda.setLike( edgeInversePermutation ) );
   edgeInversePermutationCuda = edgeInversePermutation;
   PermutationCuda cellInversePermutationCuda;
   ASSERT_TRUE( cellInversePermutationCuda.setLike( cellInversePermutation ) );
   cellInversePermutationCuda = cellInversePermutation;

   ASSERT_TRUE( meshCuda.template reorderEntities< 0 >( vertexPermutationCuda, vertexInversePermutationCuda ) );
   ASSERT_TRUE( meshCuda.template reorderEntities< 1 >( edgePermutationCuda, edgeInversePermutationCuda ) );
   ASSERT_TRUE( meshCuda.template reorderEntities< 2 >( cellPermutationCuda, cellInversePermutationCuda ) );

   EXPECT_EQ( meshCuda, mesh );
#endif
};

} // namespace MeshOrderingTest

#endif
