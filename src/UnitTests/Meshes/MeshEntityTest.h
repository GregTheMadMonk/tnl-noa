#pragma once

#ifdef HAVE_GTEST
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/Topologies/MeshEdgeTopology.h>
#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/Topologies/MeshTetrahedronTopology.h>
 
using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

using TestVertexMeshConfig = MeshConfigBase< MeshVertexTopology, 2, RealType, IndexType, IndexType, void >;
using TestEdgeMeshConfig   = MeshConfigBase< MeshEdgeTopology,   2, RealType, IndexType, IndexType, void >;

class TestTriangleMeshConfig : public MeshConfigBase< MeshTriangleTopology >
{
   public:
 
      template< typename MeshEntity >
      static constexpr bool subentityStorage( MeshEntity entity, int subentityDimensions )
      {
         return true;
      }
 
      template< typename MeshEntity >
      static constexpr bool superentityStorage( MeshEntity entity, int superentityDimensions )
      {
         return true;
      }
};

class TestTetrahedronMeshConfig : public MeshConfigBase< MeshTetrahedronTopology >
{
   public:
 
      template< typename MeshEntity >
      static constexpr bool subentityStorage( MeshEntity entity, int subentityDimensions )
      {
         return true;
      }
 
      template< typename MeshEntity >
      static constexpr bool superentityStorage( MeshEntity entity, int superentityDimensions )
      {
         return true;
      }
};

template< typename MeshConfig, typename EntityTopology, int Dimensions >
using StorageNetwork = typename MeshSuperentityTraits< MeshConfig, EntityTopology, Dimensions >::StorageNetworkType;

// stupid wrapper around MeshEntity to expose protected members needed for tests
template< typename MeshConfig, typename EntityTopology >
class TestMeshEntity
   : public MeshEntity< MeshConfig, EntityTopology >
{
   using BaseType = MeshEntity< MeshConfig, EntityTopology >;

public:
   template< int Subdimensions >
   void setSubentityIndex( const typename BaseType::LocalIndexType& localIndex,
                           const typename BaseType::GlobalIndexType& globalIndex )
   {
      BaseType::template setSubentityIndex< Subdimensions >( localIndex, globalIndex );
   }

   using BaseType::bindSuperentitiesStorageNetwork;
   using BaseType::setNumberOfSuperentities;
   using BaseType::setSuperentityIndex;
};
 
TEST( MeshEntityTest, VertexMeshEntityTest )
{
   typedef TestMeshEntity< TestVertexMeshConfig, MeshVertexTopology > VertexMeshEntityType;
   typedef typename VertexMeshEntityType::PointType PointType;

   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );
   VertexMeshEntityType vertexEntity;
   PointType point;

   point.x() = 1.0;
   point.y() = 2.0;
   vertexEntity.setPoint( point );
   ASSERT_TRUE( vertexEntity.getPoint() == point );
}

TEST( MeshEntityTest, EdgeMeshEntityTest )
{
   typedef TestMeshEntity< TestVertexMeshConfig, MeshVertexTopology > VertexMeshEntityType;
   typedef TestMeshEntity< TestEdgeMeshConfig, MeshEdgeTopology > EdgeMeshEntityType;

   typedef typename VertexMeshEntityType::PointType PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

   ASSERT_TRUE( EdgeMeshEntityType().template subentitiesAvailable< 0 >() );

   /****
    *
    * Here we test the following simple example:
    *

             point2
                |\
                | \
                |  \
                |   \

                  ....
             edge1     edge0
                  ....


                |                 \
                |                  \
                ---------------------
             point0   edge2        point1

    */

   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 );

   Containers::StaticArray< 3, VertexMeshEntityType > vertexEntities;
   vertexEntities[ 0 ].setPoint( point0 );
   vertexEntities[ 1 ].setPoint( point1 );
   vertexEntities[ 2 ].setPoint( point2 );

   ASSERT_TRUE( vertexEntities[ 0 ].getPoint() == point0 );
   ASSERT_TRUE( vertexEntities[ 1 ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ 2 ].getPoint() == point2 );

   Containers::StaticArray< 3, EdgeMeshEntityType > edgeEntities;
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, 0 );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, 1 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, 1 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, 2 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, 0 );
   edgeEntities[ 0 ].setId( 0 );
   edgeEntities[ 1 ].setId( 1 );
   edgeEntities[ 2 ].setId( 2 );

   ASSERT_TRUE( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 0 ) ].getPoint() == point0 );
   ASSERT_TRUE( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 1 ) ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 0 ) ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 1 ) ].getPoint() == point2 );
   ASSERT_TRUE( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 0 ) ].getPoint() == point2 );
   ASSERT_TRUE( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 1 ) ].getPoint() == point0 );
}

TEST( MeshEntityTest, TriangleMeshEntityTest )
{
   typedef TestMeshEntity< TestVertexMeshConfig, MeshVertexTopology > VertexMeshEntityType;
   typedef TestMeshEntity< TestEdgeMeshConfig, MeshEdgeTopology > EdgeMeshEntityType;
   typedef TestMeshEntity< TestTriangleMeshConfig, MeshTriangleTopology > TriangleMeshEntityType;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangular mesh does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "" );
   typedef typename VertexMeshEntityType::PointType PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 2, RealType >::getType() ) );

   /****
    * We set-up the same situation as in the test above
    */
   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 );

   Containers::StaticArray< 3, VertexMeshEntityType > vertexEntities;
   vertexEntities[ 0 ].setPoint( point0 );
   vertexEntities[ 1 ].setPoint( point1 );
   vertexEntities[ 2 ].setPoint( point2 );

   ASSERT_TRUE( vertexEntities[ 0 ].getPoint() == point0 );
   ASSERT_TRUE( vertexEntities[ 1 ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ 2 ].getPoint() == point2 );

   Containers::StaticArray< 3, EdgeMeshEntityType > edgeEntities;
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 0 >::index );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 1 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 0 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 1 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 0 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 1 >::index );

   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 1 >::index ) );

   TriangleMeshEntityType triangleEntity;

   triangleEntity.template setSubentityIndex< 0 >( 0 , 0 );
   triangleEntity.template setSubentityIndex< 0 >( 1 , 1 );
   triangleEntity.template setSubentityIndex< 0 >( 2 , 2 );

   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 0 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 0 >( 1 ) == 1 );
   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 0 >( 2 ) == 2 );

   triangleEntity.template setSubentityIndex< 1 >( 0 , 0 );
   triangleEntity.template setSubentityIndex< 1 >( 1 , 1 );
   triangleEntity.template setSubentityIndex< 1 >( 2 , 2 );

   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 1 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 1 >( 1 ) == 1 );
   ASSERT_TRUE( triangleEntity.template getSubentityIndex< 1 >( 2 ) == 2 );
}

TEST( MeshEntityTest, TetragedronMeshEntityTest )
{
   //typedef MeshConfigBase< MeshTetrahedronTopology, 3, RealType, IndexType, IndexType, void > TestTetrahedronEntityTopology;
   typedef MeshConfigBase< MeshTriangleTopology, 3, RealType, IndexType, IndexType, void > TestTriangleEntityTopology;
   typedef MeshConfigBase< MeshEdgeTopology, 3, RealType, IndexType, IndexType, void > TestEdgeEntityTopology;
   typedef MeshConfigBase< MeshVertexTopology, 3, RealType, IndexType, IndexType, void > TestVertexEntityTopology;

   typedef TestMeshEntity< TestTetrahedronMeshConfig, MeshTetrahedronTopology > TetrahedronMeshEntityType;
   typedef TestMeshEntity< TestTriangleMeshConfig, MeshTriangleTopology > TriangleMeshEntityType;
   typedef TestMeshEntity< TestEdgeEntityTopology, MeshEdgeTopology > EdgeMeshEntityType;
   typedef TestMeshEntity< TestVertexEntityTopology, MeshVertexTopology > VertexMeshEntityType;
   typedef typename VertexMeshEntityType::PointType PointType;
   ASSERT_TRUE( PointType::getType() == ( Containers::StaticVector< 3, RealType >::getType() ) );

   /****
    * We set-up similar situation as above but with
    * tetrahedron.
    */
   PointType point0( 0.0, 0.0, 0.0),
             point1( 1.0, 0.0, 0.0 ),
             point2( 0.0, 1.0, 0.0 ),
             point3( 0.0, 0.0, 1.0 );

   Containers::StaticArray< MeshSubtopology< MeshTetrahedronTopology, 0 >::count,
                   VertexMeshEntityType > vertexEntities;

   vertexEntities[ 0 ].setPoint( point0 );
   vertexEntities[ 1 ].setPoint( point1 );
   vertexEntities[ 2 ].setPoint( point2 );
   vertexEntities[ 3 ].setPoint( point3 );

   ASSERT_TRUE( vertexEntities[ 0 ].getPoint() == point0 );
   ASSERT_TRUE( vertexEntities[ 1 ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ 2 ].getPoint() == point2 );
   ASSERT_TRUE( vertexEntities[ 3 ].getPoint() == point3 );

   Containers::StaticArray< MeshSubtopology< MeshTetrahedronTopology, 1 >::count,
                   EdgeMeshEntityType > edgeEntities;
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 0 >::index );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 1 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 0 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 1 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 0 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 1 >::index );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 0 >::index );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 1 >::index );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 0 >::index );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 1 >::index );
   edgeEntities[ 5 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 0 >::index );
   edgeEntities[ 5 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 1 >::index );

   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 3 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 3 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 4 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 4 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 1 >::index ) );
   ASSERT_TRUE( edgeEntities[ 5 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 0 >::index ) );
   ASSERT_TRUE( edgeEntities[ 5 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 1 >::index ) );

   Containers::StaticArray< MeshSubtopology< MeshTetrahedronTopology, 2 >::count,
                   TriangleMeshEntityType > triangleEntities;
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 0 >::index );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 1 >::index );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 2, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 2 >::index );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 0 >::index );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 1 >::index );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 2, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 2 >::index );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 0 >::index );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 1 >::index );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 2, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 2 >::index );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 0, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 0 >::index );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 1, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 1 >::index );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 2, tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 2 >::index );

   ASSERT_TRUE( triangleEntities[ 0 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 0 >::index ) );
   ASSERT_TRUE( triangleEntities[ 0 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 1 >::index ) );
   ASSERT_TRUE( triangleEntities[ 0 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 2 >::index ) );
   ASSERT_TRUE( triangleEntities[ 1 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 0 >::index ) );
   ASSERT_TRUE( triangleEntities[ 1 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 1 >::index ) );
   ASSERT_TRUE( triangleEntities[ 1 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 2 >::index ) );
   ASSERT_TRUE( triangleEntities[ 2 ].getVertexIndex( 0 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 0 >::index ) );
   ASSERT_TRUE( triangleEntities[ 2 ].getVertexIndex( 1 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 1 >::index ) );
   ASSERT_TRUE( triangleEntities[ 2 ].getVertexIndex( 2 ) == ( tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 2 >::index ) );

   TetrahedronMeshEntityType tetrahedronEntity;
   tetrahedronEntity.template setSubentityIndex< 0 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 3, 3 );

   ASSERT_TRUE( tetrahedronEntity.getVertexIndex( 0 ) == 0 );
   ASSERT_TRUE( tetrahedronEntity.getVertexIndex( 1 ) == 1 );
   ASSERT_TRUE( tetrahedronEntity.getVertexIndex( 2 ) == 2 );
   ASSERT_TRUE( tetrahedronEntity.getVertexIndex( 3 ) == 3 );

   tetrahedronEntity.template setSubentityIndex< 2 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 3, 3 );

   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 2 >( 0 ) == 0 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 2 >( 1 ) == 1 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 2 >( 2 ) == 2 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 2 >( 3 ) == 3 );

   tetrahedronEntity.template setSubentityIndex< 1 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 3, 3 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 4, 4 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 5, 5 );

   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 0 ) == 0 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 1 ) == 1 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 2 ) == 2 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 3 ) == 3 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 4 ) == 4 );
   ASSERT_TRUE( tetrahedronEntity.template getSubentityIndex< 1 >( 5 ) == 5 );
}

TEST( MeshEntityTest, TwoTrianglesMeshEntityTest )
{
   typedef TestMeshEntity< TestTriangleMeshConfig, MeshTriangleTopology > TriangleMeshEntityType;
   typedef TestMeshEntity< TestTriangleMeshConfig, MeshEdgeTopology > EdgeMeshEntityType;
   typedef TestMeshEntity< TestTriangleMeshConfig, MeshVertexTopology > VertexMeshEntityType;
   typedef typename VertexMeshEntityType::PointType PointType;
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

   Containers::StaticArray< 4, VertexMeshEntityType > vertexEntities;
   vertexEntities[ 0 ].setPoint( point0 );
   vertexEntities[ 1 ].setPoint( point1 );
   vertexEntities[ 2 ].setPoint( point2 );
   vertexEntities[ 3 ].setPoint( point3 );

   ASSERT_TRUE( vertexEntities[ 0 ].getPoint() == point0 );
   ASSERT_TRUE( vertexEntities[ 1 ].getPoint() == point1 );
   ASSERT_TRUE( vertexEntities[ 2 ].getPoint() == point2 );
   ASSERT_TRUE( vertexEntities[ 3 ].getPoint() == point3 );

   Containers::StaticArray< 5, EdgeMeshEntityType > edgeEntities;
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, 1 );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, 2 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, 0 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, 0 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, 1 );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 1, 3 );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 0, 3 );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 1, 1 );

   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 0 ) == 1 );
   ASSERT_TRUE( edgeEntities[ 0 ].getVertexIndex( 1 ) == 2 );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 0 ) == 2 );
   ASSERT_TRUE( edgeEntities[ 1 ].getVertexIndex( 1 ) == 0 );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 0 ) == 0 );
   ASSERT_TRUE( edgeEntities[ 2 ].getVertexIndex( 1 ) == 1 );
   ASSERT_TRUE( edgeEntities[ 3 ].getVertexIndex( 0 ) == 2 );
   ASSERT_TRUE( edgeEntities[ 3 ].getVertexIndex( 1 ) == 3 );
   ASSERT_TRUE( edgeEntities[ 4 ].getVertexIndex( 0 ) == 3 );
   ASSERT_TRUE( edgeEntities[ 4 ].getVertexIndex( 1 ) == 1 );

   Containers::StaticArray< 2, TriangleMeshEntityType > triangleEntities;

   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 0 , 0 );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 1 , 1 );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 2 , 2 );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 0 , 0 );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 1 , 1 );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 2 , 2 );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 0 , 0 );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 1 , 2 );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 2 , 3 );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 0 , 0 );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 1 , 3 );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 2 , 4 );

   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 1 ) == 1 );
   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 2 ) == 2 );
   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 1 ) == 1 );
   ASSERT_TRUE( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 2 ) == 2 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 1 ) == 2 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 2 ) == 3 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 0 ) == 0 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 1 ) == 3 );
   ASSERT_TRUE( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 2 ) == 4 );


   /*
    * Tests for the superentities layer.
    */
   StorageNetwork< TestTriangleMeshConfig, MeshVertexTopology, 1 > vertexEdgeSuperentities;
   vertexEdgeSuperentities.setRanges( 4, 3 );
   vertexEdgeSuperentities.allocate( 3 );

   vertexEntities[ 0 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 0 ) );
   vertexEntities[ 0 ].template setNumberOfSuperentities< 1 >( 2 );
   vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 0, 2 );
   vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 1, 1 );

   ASSERT_EQ( vertexEntities[ 0 ].template getNumberOfSuperentities< 1 >(),  2 );
   ASSERT_EQ( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 0 ),    2 );
   ASSERT_EQ( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 1 ),    1 );

   vertexEntities[ 1 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 1 ) );
   vertexEntities[ 1 ].template setNumberOfSuperentities< 1 >( 3 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 0, 0 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 1, 2 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 2, 4 );

   ASSERT_EQ( vertexEntities[ 1 ].template getNumberOfSuperentities< 1 >(),  3 );
   ASSERT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 0 ),    0 );
   ASSERT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 1 ),    2 );
   ASSERT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 2 ),    4 );


   StorageNetwork< TestTriangleMeshConfig, MeshVertexTopology, 2 > vertexCellSuperentities;
   vertexCellSuperentities.setRanges( 4, 2 );
   vertexCellSuperentities.allocate( 2 );

   vertexEntities[ 1 ].template bindSuperentitiesStorageNetwork< 2 >( vertexCellSuperentities.getValues( 1 ) );
   vertexEntities[ 1 ].template setNumberOfSuperentities< 2 >( 2 );
   vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );
   vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 1, 1 );

   ASSERT_EQ( vertexEntities[ 1 ].template getNumberOfSuperentities< 2 >(),  2 );
   ASSERT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 0 ),    0 );
   ASSERT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 1 ),    1 );


   StorageNetwork< TestTriangleMeshConfig, MeshEdgeTopology, 2 > edgeCellSuperentities;
   edgeCellSuperentities.setRanges( 5, 2 );
   edgeCellSuperentities.allocate( 2 );

   edgeEntities[ 0 ].template bindSuperentitiesStorageNetwork< 2 >( edgeCellSuperentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setNumberOfSuperentities< 2 >( 2 );
   edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );
   edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 1, 1 );

   ASSERT_EQ( edgeEntities[ 0 ].template getNumberOfSuperentities< 2 >(),  2 );
   ASSERT_EQ( edgeEntities[ 0 ].template getSuperentityIndex< 2 >( 0 ),    0 );
}

#endif
