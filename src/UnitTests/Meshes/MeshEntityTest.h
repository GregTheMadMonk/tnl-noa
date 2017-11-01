#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
 
namespace MeshEntityTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

using TestEdgeMeshConfig = MeshConfigBase< Topologies::Edge,   2, RealType, IndexType, IndexType, void >;

class TestTriangleMeshConfig : public MeshConfigBase< Topologies::Triangle >
{
public:
   template< typename EntityTopology >
   static constexpr bool subentityStorage( EntityTopology entity, int subentityDimensions )
   {
      return true;
   }

   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology entity, int superentityDimensions )
   {
      return true;
   }
};

class TestTetrahedronMeshConfig : public MeshConfigBase< Topologies::Tetrahedron >
{
public:
   template< typename EntityTopology >
   static constexpr bool subentityStorage( EntityTopology entity, int subentityDimensions )
   {
      return true;
   }

   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology entity, int superentityDimensions )
   {
      return true;
   }
};

template< typename MeshConfig, typename EntityTopology, int Dimensions >
using SubentityStorage = typename MeshSubentityTraits< MeshConfig, Devices::Host, EntityTopology, Dimensions >::StorageNetworkType;

template< typename MeshConfig, typename EntityTopology, int Dimensions >
using SuperentityStorage = typename MeshSuperentityTraits< MeshConfig, Devices::Host, EntityTopology, Dimensions >::StorageNetworkType;

// stupid wrapper around MeshEntity to expose protected members needed for tests
template< typename MeshConfig, typename EntityTopology >
class TestMeshEntity
   : public MeshEntity< MeshConfig, Devices::Host, EntityTopology >
{
   using BaseType = MeshEntity< MeshConfig, Devices::Host, EntityTopology >;

public:
   template< int Subdimensions, typename Storage >
   void bindSubentitiesStorageNetwork( const Storage& storage )
   {
      BaseType::template bindSubentitiesStorageNetwork< Subdimensions >( storage );
   }

   template< int Subdimensions >
   void setSubentityIndex( const typename BaseType::LocalIndexType& localIndex,
                           const typename BaseType::GlobalIndexType& globalIndex )
   {
      BaseType::template setSubentityIndex< Subdimensions >( localIndex, globalIndex );
   }

   using BaseType::bindSuperentitiesStorageNetwork;
   using BaseType::setNumberOfSuperentities;
   using BaseType::setSuperentityIndex;
   using BaseType::setIndex;
};

template< typename Entity >
void generalTestSubentities( const Entity& entity )
{
   Entity copy1( entity );
   Entity copy2 = entity;

   // check that subentity accessors have been rebound, at least for the 0th subvertex
   EXPECT_EQ( copy1.template getSubentityIndex< 0 >( 0 ), entity.template getSubentityIndex< 0 >( 0 ) );
   EXPECT_EQ( copy2.template getSubentityIndex< 0 >( 0 ), entity.template getSubentityIndex< 0 >( 0 ) );
}
 
template< typename Entity >
void generalTestSuperentities( const Entity& entity )
{
   Entity copy1( entity );
   Entity copy2 = entity;

   // check that subentity accessors have been rebound, at least for the 0th superentity
   EXPECT_EQ( copy1.template getSuperentityIndex< Entity::getEntityDimension() + 1 >( 0 ), entity.template getSuperentityIndex< Entity::getEntityDimension() + 1 >( 0 ) );
   EXPECT_EQ( copy2.template getSuperentityIndex< Entity::getEntityDimension() + 1 >( 0 ), entity.template getSuperentityIndex< Entity::getEntityDimension() + 1 >( 0 ) );
}
 
TEST( MeshEntityTest, VertexMeshEntityTest )
{
   using EdgeMeshEntityType = TestMeshEntity< TestEdgeMeshConfig, Topologies::Edge >;
   using VertexMeshEntityType = TestMeshEntity< TestEdgeMeshConfig, typename EdgeMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(),  ( Containers::StaticVector< 2, RealType >::getType() ) );

   VertexMeshEntityType vertexEntity;
   PointType point;
   point.x() = 1.0;
   point.y() = 2.0;
   vertexEntity.setPoint( point );
   EXPECT_EQ( vertexEntity.getPoint(),  point );
}

TEST( MeshEntityTest, EdgeMeshEntityTest )
{
   using EdgeMeshEntityType = TestMeshEntity< TestEdgeMeshConfig, Topologies::Edge >;
   using VertexMeshEntityType = TestMeshEntity< TestEdgeMeshConfig, typename EdgeMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(),  ( Containers::StaticVector< 2, RealType >::getType() ) );

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

   EXPECT_EQ( vertexEntities[ 0 ].getPoint(), point0 );
   EXPECT_EQ( vertexEntities[ 1 ].getPoint(), point1 );
   EXPECT_EQ( vertexEntities[ 2 ].getPoint(), point2 );

   Containers::StaticArray< 3, EdgeMeshEntityType > edgeEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Edge, 0 > edgeVertexSubentities;
   edgeVertexSubentities.setKeysRange( 3 );
   edgeVertexSubentities.allocate();

   edgeEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, 0 );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, 1 );
   edgeEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 1 ) );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, 1 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, 2 );
   edgeEntities[ 2 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 2 ) );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, 0 );
   edgeEntities[ 0 ].setIndex( 0 );
   edgeEntities[ 1 ].setIndex( 1 );
   edgeEntities[ 2 ].setIndex( 2 );

   EXPECT_EQ( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 0 ) ].getPoint(), point0 );
   EXPECT_EQ( vertexEntities[ edgeEntities[ 0 ].getVertexIndex( 1 ) ].getPoint(), point1 );
   EXPECT_EQ( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 0 ) ].getPoint(), point1 );
   EXPECT_EQ( vertexEntities[ edgeEntities[ 1 ].getVertexIndex( 1 ) ].getPoint(), point2 );
   EXPECT_EQ( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 0 ) ].getPoint(), point2 );
   EXPECT_EQ( vertexEntities[ edgeEntities[ 2 ].getVertexIndex( 1 ) ].getPoint(), point0 );


   generalTestSubentities( edgeEntities[ 0 ] );
   generalTestSubentities( edgeEntities[ 1 ] );
   generalTestSubentities( edgeEntities[ 2 ] );
}

TEST( MeshEntityTest, TriangleMeshEntityTest )
{
   using TriangleMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, Topologies::Triangle >;
   using EdgeMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityTopology >;
   using VertexMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(), ( Containers::StaticVector< 2, RealType >::getType() ) );

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

   EXPECT_EQ( vertexEntities[ 0 ].getPoint(), point0 );
   EXPECT_EQ( vertexEntities[ 1 ].getPoint(), point1 );
   EXPECT_EQ( vertexEntities[ 2 ].getPoint(), point2 );

   Containers::StaticArray< 3, EdgeMeshEntityType > edgeEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Edge, 0 > edgeVertexSubentities;
   edgeVertexSubentities.setKeysRange( 3 );
   edgeVertexSubentities.allocate();

   edgeEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 0, 0 >::index );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 0, 1 >::index );
   edgeEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 1 ) );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 1, 0 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 1, 1 >::index );
   edgeEntities[ 2 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 2 ) );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 2, 0 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 2, 1 >::index );

   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 0 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 0, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 1 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 0, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 0 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 1, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 1 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 1, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 0 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 2, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 1 ), ( Topologies::SubentityVertexMap< Topologies::Triangle, Topologies::Edge, 2, 1 >::index ) );

   TriangleMeshEntityType triangleEntity;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 0 > triangleVertexSubentities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 1 > triangleEdgeSubentities;
   triangleVertexSubentities.setKeysRange( 1 );
   triangleEdgeSubentities.setKeysRange( 1 );
   triangleVertexSubentities.allocate();
   triangleEdgeSubentities.allocate();

   triangleEntity.template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 0 ) );
   triangleEntity.template setSubentityIndex< 0 >( 0 , 0 );
   triangleEntity.template setSubentityIndex< 0 >( 1 , 1 );
   triangleEntity.template setSubentityIndex< 0 >( 2 , 2 );

   EXPECT_EQ( triangleEntity.template getSubentityIndex< 0 >( 0 ), 0 );
   EXPECT_EQ( triangleEntity.template getSubentityIndex< 0 >( 1 ), 1 );
   EXPECT_EQ( triangleEntity.template getSubentityIndex< 0 >( 2 ), 2 );

   triangleEntity.template bindSubentitiesStorageNetwork< 1 >( triangleEdgeSubentities.getValues( 0 ) );
   triangleEntity.template setSubentityIndex< 1 >( 0 , 0 );
   triangleEntity.template setSubentityIndex< 1 >( 1 , 1 );
   triangleEntity.template setSubentityIndex< 1 >( 2 , 2 );

   EXPECT_EQ( triangleEntity.template getSubentityIndex< 1 >( 0 ), 0 );
   EXPECT_EQ( triangleEntity.template getSubentityIndex< 1 >( 1 ), 1 );
   EXPECT_EQ( triangleEntity.template getSubentityIndex< 1 >( 2 ), 2 );
}

TEST( MeshEntityTest, TetrahedronMeshEntityTest )
{
   using TetrahedronMeshEntityType = TestMeshEntity< TestTetrahedronMeshConfig, Topologies::Tetrahedron >;
   using TriangleMeshEntityType = TestMeshEntity< TestTetrahedronMeshConfig, typename TetrahedronMeshEntityType::SubentityTraits< 2 >::SubentityTopology >;
   using EdgeMeshEntityType = TestMeshEntity< TestTetrahedronMeshConfig, typename TetrahedronMeshEntityType::SubentityTraits< 1 >::SubentityTopology >;
   using VertexMeshEntityType = TestMeshEntity< TestTetrahedronMeshConfig, typename TetrahedronMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;

   static_assert( TetrahedronMeshEntityType::SubentityTraits< 2 >::storageEnabled, "Testing tetrahedron entity does not store triangles as required." );
   static_assert( TetrahedronMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing tetrahedron entity does not store edges as required." );
   static_assert( TetrahedronMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing tetrahedron entity does not store vertices as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(),  ( Containers::StaticVector< 3, RealType >::getType() ) );

   /****
    * We set-up similar situation as above but with
    * tetrahedron.
    */
   PointType point0( 0.0, 0.0, 0.0),
             point1( 1.0, 0.0, 0.0 ),
             point2( 0.0, 1.0, 0.0 ),
             point3( 0.0, 0.0, 1.0 );

   Containers::StaticArray< Topologies::Subtopology< Topologies::Tetrahedron, 0 >::count,
                   VertexMeshEntityType > vertexEntities;

   vertexEntities[ 0 ].setPoint( point0 );
   vertexEntities[ 1 ].setPoint( point1 );
   vertexEntities[ 2 ].setPoint( point2 );
   vertexEntities[ 3 ].setPoint( point3 );

   EXPECT_EQ( vertexEntities[ 0 ].getPoint(),  point0 );
   EXPECT_EQ( vertexEntities[ 1 ].getPoint(),  point1 );
   EXPECT_EQ( vertexEntities[ 2 ].getPoint(),  point2 );
   EXPECT_EQ( vertexEntities[ 3 ].getPoint(),  point3 );

   Containers::StaticArray< Topologies::Subtopology< Topologies::Tetrahedron, 1 >::count,
                            EdgeMeshEntityType > edgeEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Edge, 0 > edgeVertexSubentities;
   edgeVertexSubentities.setKeysRange( 6 );
   edgeVertexSubentities.allocate();

   edgeEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 0, 0 >::index );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 0, 1 >::index );
   edgeEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 1 ) );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 1, 0 >::index );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 1, 1 >::index );
   edgeEntities[ 2 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 2 ) );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 2, 0 >::index );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 2, 1 >::index );
   edgeEntities[ 3 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 3 ) );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 3, 0 >::index );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 3, 1 >::index );
   edgeEntities[ 4 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 4 ) );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 4, 0 >::index );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 4, 1 >::index );
   edgeEntities[ 5 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 5 ) );
   edgeEntities[ 5 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 5, 0 >::index );
   edgeEntities[ 5 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 5, 1 >::index );

   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 0, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 0, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 1, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 1, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 2, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 2, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 3 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 3, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 3 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 3, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 4 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 4, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 4 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 4, 1 >::index ) );
   EXPECT_EQ( edgeEntities[ 5 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 5, 0 >::index ) );
   EXPECT_EQ( edgeEntities[ 5 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Edge, 5, 1 >::index ) );

   Containers::StaticArray< Topologies::Subtopology< Topologies::Tetrahedron, 2 >::count,
                            TriangleMeshEntityType > triangleEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 0 > triangleVertexSubentities;
   triangleVertexSubentities.setKeysRange( 4 );
   triangleVertexSubentities.allocate();

   triangleEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 0 ) );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 0 >::index );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 1 >::index );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 2, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 2 >::index );
   triangleEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 1 ) );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 0 >::index );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 1 >::index );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 2, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 2 >::index );
   triangleEntities[ 2 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 2 ) );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 0 >::index );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 1 >::index );
   triangleEntities[ 2 ].template setSubentityIndex< 0 >( 2, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 2 >::index );
   triangleEntities[ 3 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 3 ) );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 0, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 3, 0 >::index );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 1, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 3, 1 >::index );
   triangleEntities[ 3 ].template setSubentityIndex< 0 >( 2, Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 3, 2 >::index );

   EXPECT_EQ( triangleEntities[ 0 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 0 >::index ) );
   EXPECT_EQ( triangleEntities[ 0 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 1 >::index ) );
   EXPECT_EQ( triangleEntities[ 0 ].getVertexIndex( 2 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 0, 2 >::index ) );
   EXPECT_EQ( triangleEntities[ 1 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 0 >::index ) );
   EXPECT_EQ( triangleEntities[ 1 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 1 >::index ) );
   EXPECT_EQ( triangleEntities[ 1 ].getVertexIndex( 2 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 1, 2 >::index ) );
   EXPECT_EQ( triangleEntities[ 2 ].getVertexIndex( 0 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 0 >::index ) );
   EXPECT_EQ( triangleEntities[ 2 ].getVertexIndex( 1 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 1 >::index ) );
   EXPECT_EQ( triangleEntities[ 2 ].getVertexIndex( 2 ),  ( Topologies::SubentityVertexMap< Topologies::Tetrahedron, Topologies::Triangle, 2, 2 >::index ) );

   TetrahedronMeshEntityType tetrahedronEntity;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Tetrahedron, 0 > tetrahedronVertexSubentities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Tetrahedron, 1 > tetrahedronEdgeSubentities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Tetrahedron, 2 > tetrahedronTriangleSubentities;
   tetrahedronVertexSubentities.setKeysRange( 1 );
   tetrahedronEdgeSubentities.setKeysRange( 1 );
   tetrahedronTriangleSubentities.setKeysRange( 1 );
   tetrahedronVertexSubentities.allocate();
   tetrahedronEdgeSubentities.allocate();
   tetrahedronTriangleSubentities.allocate();

   tetrahedronEntity.template bindSubentitiesStorageNetwork< 0 >( tetrahedronVertexSubentities.getValues( 0 ) );
   tetrahedronEntity.template setSubentityIndex< 0 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 0 >( 3, 3 );

   EXPECT_EQ( tetrahedronEntity.getVertexIndex( 0 ),  0 );
   EXPECT_EQ( tetrahedronEntity.getVertexIndex( 1 ),  1 );
   EXPECT_EQ( tetrahedronEntity.getVertexIndex( 2 ),  2 );
   EXPECT_EQ( tetrahedronEntity.getVertexIndex( 3 ),  3 );

   tetrahedronEntity.template bindSubentitiesStorageNetwork< 2 >( tetrahedronTriangleSubentities.getValues( 0 ) );
   tetrahedronEntity.template setSubentityIndex< 2 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 2 >( 3, 3 );

   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 2 >( 0 ),  0 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 2 >( 1 ),  1 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 2 >( 2 ),  2 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 2 >( 3 ),  3 );

   tetrahedronEntity.template bindSubentitiesStorageNetwork< 1 >( tetrahedronEdgeSubentities.getValues( 0 ) );
   tetrahedronEntity.template setSubentityIndex< 1 >( 0, 0 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 1, 1 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 2, 2 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 3, 3 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 4, 4 );
   tetrahedronEntity.template setSubentityIndex< 1 >( 5, 5 );

   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 3 ),  3 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 4 ),  4 );
   EXPECT_EQ( tetrahedronEntity.template getSubentityIndex< 1 >( 5 ),  5 );


   generalTestSubentities( edgeEntities[ 0 ] );
   generalTestSubentities( edgeEntities[ 1 ] );
   generalTestSubentities( edgeEntities[ 2 ] );
   generalTestSubentities( tetrahedronEntity );
}

TEST( MeshEntityTest, TwoTrianglesMeshEntityTest )
{
   using TriangleMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, Topologies::Triangle >;
   using EdgeMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityTopology >;
   using VertexMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(),  ( Containers::StaticVector< 2, RealType >::getType() ) );

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

   EXPECT_EQ( vertexEntities[ 0 ].getPoint(),  point0 );
   EXPECT_EQ( vertexEntities[ 1 ].getPoint(),  point1 );
   EXPECT_EQ( vertexEntities[ 2 ].getPoint(),  point2 );
   EXPECT_EQ( vertexEntities[ 3 ].getPoint(),  point3 );

   Containers::StaticArray< 5, EdgeMeshEntityType > edgeEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Edge, 0 > edgeVertexSubentities;
   edgeVertexSubentities.setKeysRange( 5 );
   edgeVertexSubentities.allocate();

   edgeEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 0, 1 );
   edgeEntities[ 0 ].template setSubentityIndex< 0 >( 1, 2 );
   edgeEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 1 ) );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 1 ].template setSubentityIndex< 0 >( 1, 0 );
   edgeEntities[ 2 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 2 ) );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 0, 0 );
   edgeEntities[ 2 ].template setSubentityIndex< 0 >( 1, 1 );
   edgeEntities[ 3 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 3 ) );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 0, 2 );
   edgeEntities[ 3 ].template setSubentityIndex< 0 >( 1, 3 );
   edgeEntities[ 4 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 4 ) );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 0, 3 );
   edgeEntities[ 4 ].template setSubentityIndex< 0 >( 1, 1 );

   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 0 ),  1 );
   EXPECT_EQ( edgeEntities[ 0 ].getVertexIndex( 1 ),  2 );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 0 ),  2 );
   EXPECT_EQ( edgeEntities[ 1 ].getVertexIndex( 1 ),  0 );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 0 ),  0 );
   EXPECT_EQ( edgeEntities[ 2 ].getVertexIndex( 1 ),  1 );
   EXPECT_EQ( edgeEntities[ 3 ].getVertexIndex( 0 ),  2 );
   EXPECT_EQ( edgeEntities[ 3 ].getVertexIndex( 1 ),  3 );
   EXPECT_EQ( edgeEntities[ 4 ].getVertexIndex( 0 ),  3 );
   EXPECT_EQ( edgeEntities[ 4 ].getVertexIndex( 1 ),  1 );

   Containers::StaticArray< 2, TriangleMeshEntityType > triangleEntities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 0 > triangleVertexSubentities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 1 > triangleEdgeSubentities;
   triangleVertexSubentities.setKeysRange( 2 );
   triangleVertexSubentities.allocate();
   triangleEdgeSubentities.setKeysRange( 2 );
   triangleEdgeSubentities.allocate();

   triangleEntities[ 0 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 0 ) );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 0 , 0 );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 1 , 1 );
   triangleEntities[ 0 ].template setSubentityIndex< 0 >( 2 , 2 );
   triangleEntities[ 0 ].template bindSubentitiesStorageNetwork< 1 >( triangleEdgeSubentities.getValues( 0 ) );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 0 , 0 );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 1 , 1 );
   triangleEntities[ 0 ].template setSubentityIndex< 1 >( 2 , 2 );
   triangleEntities[ 1 ].template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 1 ) );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 0 , 1 );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 1 , 2 );
   triangleEntities[ 1 ].template setSubentityIndex< 0 >( 2 , 3 );
   triangleEntities[ 1 ].template bindSubentitiesStorageNetwork< 1 >( triangleEdgeSubentities.getValues( 1 ) );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 0 , 3 );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 1 , 4 );
   triangleEntities[ 1 ].template setSubentityIndex< 1 >( 2 , 0 );

   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( triangleEntities[ 0 ].template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 0 >( 2 ),  3 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 0 ),  3 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 1 ),  4 );
   EXPECT_EQ( triangleEntities[ 1 ].template getSubentityIndex< 1 >( 2 ),  0 );


   /*
    * Tests for the superentities layer.
    */
   SuperentityStorage< TestTriangleMeshConfig, Topologies::Vertex, 1 > vertexEdgeSuperentities;
   vertexEdgeSuperentities.setKeysRange( 4 );
   vertexEdgeSuperentities.allocate( 3 );

   vertexEntities[ 0 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 0 ) );
   vertexEntities[ 0 ].template setNumberOfSuperentities< 1 >( 2 );
   vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 0, 2 );
   vertexEntities[ 0 ].template setSuperentityIndex< 1 >( 1, 1 );

   EXPECT_EQ( vertexEntities[ 0 ].template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 0 ),    2 );
   EXPECT_EQ( vertexEntities[ 0 ].template getSuperentityIndex< 1 >( 1 ),    1 );

   vertexEntities[ 1 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 1 ) );
   vertexEntities[ 1 ].template setNumberOfSuperentities< 1 >( 3 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 0, 0 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 1, 2 );
   vertexEntities[ 1 ].template setSuperentityIndex< 1 >( 2, 4 );

   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 0 ),    0 );
   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 1 ),    2 );
   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 1 >( 2 ),    4 );


   SuperentityStorage< TestTriangleMeshConfig, Topologies::Vertex, 2 > vertexCellSuperentities;
   vertexCellSuperentities.setKeysRange( 4 );
   vertexCellSuperentities.allocate( 2 );

   vertexEntities[ 1 ].template bindSuperentitiesStorageNetwork< 2 >( vertexCellSuperentities.getValues( 1 ) );
   vertexEntities[ 1 ].template setNumberOfSuperentities< 2 >( 2 );
   vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );
   vertexEntities[ 1 ].template setSuperentityIndex< 2 >( 1, 1 );

   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( vertexEntities[ 1 ].template getSuperentityIndex< 2 >( 1 ),    1 );


   SuperentityStorage< TestTriangleMeshConfig, Topologies::Edge, 2 > edgeCellSuperentities;
   edgeCellSuperentities.setKeysRange( 5 );
   edgeCellSuperentities.allocate( 2 );

   edgeEntities[ 0 ].template bindSuperentitiesStorageNetwork< 2 >( edgeCellSuperentities.getValues( 0 ) );
   edgeEntities[ 0 ].template setNumberOfSuperentities< 2 >( 2 );
   edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );
   edgeEntities[ 0 ].template setSuperentityIndex< 2 >( 1, 1 );

   EXPECT_EQ( edgeEntities[ 0 ].template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( edgeEntities[ 0 ].template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( edgeEntities[ 0 ].template getSuperentityIndex< 2 >( 1 ),    1 );


   generalTestSuperentities( vertexEntities[ 0 ] );
   generalTestSuperentities( vertexEntities[ 1 ] );
   generalTestSuperentities( edgeEntities[ 0 ] );
   generalTestSubentities( edgeEntities[ 0 ] );
   generalTestSubentities( edgeEntities[ 1 ] );
   generalTestSubentities( edgeEntities[ 2 ] );
   generalTestSubentities( edgeEntities[ 3 ] );
   generalTestSubentities( edgeEntities[ 4 ] );
   generalTestSubentities( triangleEntities[ 0 ] );
   generalTestSubentities( triangleEntities[ 1 ] );
}

TEST( MeshEntityTest, OneTriangleComparisonTest )
{
   using TriangleMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, Topologies::Triangle >;
   using EdgeMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityTopology >;
   using VertexMeshEntityType = TestMeshEntity< TestTriangleMeshConfig, typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityTopology >;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   EXPECT_EQ( PointType::getType(),  ( Containers::StaticVector< 2, RealType >::getType() ) );

   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 );

   Containers::StaticArray< 3, VertexMeshEntityType > vertices;
   vertices[ 0 ].setPoint( point0 );
   vertices[ 1 ].setPoint( point1 );
   vertices[ 2 ].setPoint( point2 );

   Containers::StaticArray< 3, EdgeMeshEntityType > edges;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Edge, 0 > edgeVertexSubentities;
   edgeVertexSubentities.setKeysRange( 3 );
   edgeVertexSubentities.allocate();

   edges[ 0 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 0 ) );
   edges[ 0 ].template setSubentityIndex< 0 >( 0, 1 );
   edges[ 0 ].template setSubentityIndex< 0 >( 1, 2 );
   edges[ 1 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 1 ) );
   edges[ 1 ].template setSubentityIndex< 0 >( 0, 2 );
   edges[ 1 ].template setSubentityIndex< 0 >( 1, 0 );
   edges[ 2 ].template bindSubentitiesStorageNetwork< 0 >( edgeVertexSubentities.getValues( 2 ) );
   edges[ 2 ].template setSubentityIndex< 0 >( 0, 0 );
   edges[ 2 ].template setSubentityIndex< 0 >( 1, 1 );

   TriangleMeshEntityType triangle;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 0 > triangleVertexSubentities;
   SubentityStorage< TestTriangleMeshConfig, Topologies::Triangle, 1 > triangleEdgeSubentities;
   triangleVertexSubentities.setKeysRange( 1 );
   triangleEdgeSubentities.setKeysRange( 1 );
   triangleVertexSubentities.allocate();
   triangleEdgeSubentities.allocate();

   triangle.template bindSubentitiesStorageNetwork< 0 >( triangleVertexSubentities.getValues( 0 ) );
   triangle.template setSubentityIndex< 0 >( 0 , 0 );
   triangle.template setSubentityIndex< 0 >( 1 , 1 );
   triangle.template setSubentityIndex< 0 >( 2 , 2 );
   triangle.template bindSubentitiesStorageNetwork< 1 >( triangleVertexSubentities.getValues( 0 ) );
   triangle.template setSubentityIndex< 1 >( 0 , 0 );
   triangle.template setSubentityIndex< 1 >( 1 , 1 );
   triangle.template setSubentityIndex< 1 >( 2 , 2 );


   SuperentityStorage< TestTriangleMeshConfig, Topologies::Vertex, 1 > vertexEdgeSuperentities;
   vertexEdgeSuperentities.setKeysRange( 3 );
   vertexEdgeSuperentities.allocate( 2 );

   vertices[ 0 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 0 ) );
   vertices[ 0 ].template setNumberOfSuperentities< 1 >( 2 );
   vertices[ 0 ].template setSuperentityIndex< 1 >( 0, 2 );
   vertices[ 0 ].template setSuperentityIndex< 1 >( 1, 1 );

   vertices[ 1 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 1 ) );
   vertices[ 1 ].template setNumberOfSuperentities< 1 >( 2 );
   vertices[ 1 ].template setSuperentityIndex< 1 >( 0, 0 );
   vertices[ 1 ].template setSuperentityIndex< 1 >( 1, 2 );

   vertices[ 2 ].template bindSuperentitiesStorageNetwork< 1 >( vertexEdgeSuperentities.getValues( 2 ) );
   vertices[ 2 ].template setNumberOfSuperentities< 1 >( 2 );
   vertices[ 2 ].template setSuperentityIndex< 1 >( 0, 0 );
   vertices[ 2 ].template setSuperentityIndex< 1 >( 1, 1 );


   SuperentityStorage< TestTriangleMeshConfig, Topologies::Vertex, 2 > vertexCellSuperentities;
   vertexCellSuperentities.setKeysRange( 3 );
   vertexCellSuperentities.allocate( 1 );

   vertices[ 0 ].template bindSuperentitiesStorageNetwork< 2 >( vertexCellSuperentities.getValues( 0 ) );
   vertices[ 0 ].template setNumberOfSuperentities< 2 >( 1 );
   vertices[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );

   vertices[ 1 ].template bindSuperentitiesStorageNetwork< 2 >( vertexCellSuperentities.getValues( 1 ) );
   vertices[ 1 ].template setNumberOfSuperentities< 2 >( 1 );
   vertices[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );

   vertices[ 2 ].template bindSuperentitiesStorageNetwork< 2 >( vertexCellSuperentities.getValues( 2 ) );
   vertices[ 2 ].template setNumberOfSuperentities< 2 >( 1 );
   vertices[ 2 ].template setSuperentityIndex< 2 >( 0, 0 );


   SuperentityStorage< TestTriangleMeshConfig, Topologies::Edge, 2 > edgeCellSuperentities;
   edgeCellSuperentities.setKeysRange( 3 );
   edgeCellSuperentities.allocate( 1 );

   edges[ 0 ].template bindSuperentitiesStorageNetwork< 2 >( edgeCellSuperentities.getValues( 0 ) );
   edges[ 0 ].template setNumberOfSuperentities< 2 >( 1 );
   edges[ 0 ].template setSuperentityIndex< 2 >( 0, 0 );

   edges[ 1 ].template bindSuperentitiesStorageNetwork< 2 >( edgeCellSuperentities.getValues( 1 ) );
   edges[ 1 ].template setNumberOfSuperentities< 2 >( 1 );
   edges[ 1 ].template setSuperentityIndex< 2 >( 0, 0 );

   edges[ 2 ].template bindSuperentitiesStorageNetwork< 2 >( edgeCellSuperentities.getValues( 2 ) );
   edges[ 2 ].template setNumberOfSuperentities< 2 >( 1 );
   edges[ 2 ].template setSuperentityIndex< 2 >( 0, 0 );


   /*
    * Tests for MeshEntity::operator==
    */
   EXPECT_EQ( vertices[ 0 ], vertices[ 0 ] );
   EXPECT_NE( vertices[ 0 ], vertices[ 1 ] );
   vertices[ 0 ].setPoint( point1 );
   vertices[ 0 ].template setSuperentityIndex< 1 >( 0, 0 );
   vertices[ 0 ].template setSuperentityIndex< 1 >( 1, 2 );
   EXPECT_EQ( vertices[ 0 ], vertices[ 1 ] );
   vertices[ 0 ].template setSuperentityIndex< 2 >( 0, 1 );
   EXPECT_NE( vertices[ 0 ], vertices[ 1 ] );
   vertices[ 1 ].template setSuperentityIndex< 2 >( 0, 1 );
   EXPECT_EQ( vertices[ 0 ], vertices[ 1 ] );

   EXPECT_EQ( edges[ 0 ], edges[ 0 ] );
   EXPECT_NE( edges[ 0 ], edges[ 1 ] );
   edges[ 0 ].template setSubentityIndex< 0 >( 0, 2 );
   edges[ 0 ].template setSubentityIndex< 0 >( 1, 0 );
   EXPECT_EQ( edges[ 0 ], edges[ 1 ] );
   edges[ 0 ].template setSuperentityIndex< 2 >( 0, 1 );
   EXPECT_NE( edges[ 0 ], edges[ 1 ] );
   edges[ 1 ].template setSuperentityIndex< 2 >( 0, 1 );
   EXPECT_EQ( edges[ 0 ], edges[ 1 ] );


   /*
    * Tests for copy-assignment
    */
   VertexMeshEntityType v1( vertices[ 0 ] );
   EXPECT_EQ( v1, vertices[ 0 ] );
   VertexMeshEntityType v2 = vertices[ 0 ];
   EXPECT_EQ( v2, vertices[ 0 ] );

   EdgeMeshEntityType e1( edges[ 0 ] );
   EXPECT_EQ( e1, edges[ 0 ] );
   EdgeMeshEntityType e2 = edges[ 0 ];
   EXPECT_EQ( e2, edges[ 0 ] );

   TriangleMeshEntityType t1( triangle );
   EXPECT_EQ( t1, triangle );
   TriangleMeshEntityType t2 = triangle;
   EXPECT_EQ( t2, triangle );
}

} // namespace MeshEntityTest

#endif
