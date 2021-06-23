#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>

namespace TNL {
namespace Meshes {

enum class PolygonDecomposerVersion
{
   ConnectEdgesToCentroid, ConnectEdgesToPoint
};

template< typename MeshConfig, PolygonDecomposerVersion >
struct PolygonDecomposer;

template< typename MeshConfig >
struct PolygonDecomposer< MeshConfig, PolygonDecomposerVersion::ConnectEdgesToCentroid >
{
   using Device = typename Devices::Host;
   using Topology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;

   static GlobalIndexType getExtraPointsCount( const MeshEntityType & entity )
   {
      return 1;
   }

   static GlobalIndexType getEntitiesCount( const MeshEntityType & entity )
   {
      const auto pointsCount = entity.template getSubentitiesCount< 0 >();
      return ( pointsCount == 3 ) ? 1 : pointsCount; // polygon is decomposed only if it has more than 3 vertices
   }

   template< typename MeshType, typename EntitySeedGetterType >
   static void decompose( MeshBuilder< MeshType > & meshBuilder,
                          EntitySeedGetterType entitySeedGetter,
                          GlobalIndexType & pointsCount,
                          GlobalIndexType & entitiesCount,
                          const MeshEntityType & entity )
   {
      const auto verticesCount = entity.template getSubentitiesCount< 0 >();
      if( verticesCount == 3 ) { // polygon is not decomposed as it's already a triangle
         auto & entitySeed = entitySeedGetter( entitiesCount++ );
         entitySeed.setCornersCount( 3 );
         entitySeed.setCornerId( 0, entity.template getSubentityIndex< 0 >( 0 ) );
         entitySeed.setCornerId( 1, entity.template getSubentityIndex< 0 >( 1 ) );
         entitySeed.setCornerId( 2, entity.template getSubentityIndex< 0 >( 2 ) );
      }
      else { // polygon is decomposed as it has got more than 3 vertices
         // add centroid of entity to points
         const auto entityCenter = getEntityCenter( entity.getMesh(), entity );
         const auto entityCenterIdx = pointsCount++;
         meshBuilder.setPoint( entityCenterIdx, entityCenter );
         // decompose polygon into triangles by connecting each edge to the centroid
         for( LocalIndexType j = 0, k = 1; k < verticesCount; j++, k++ ) {
            auto & entitySeed = entitySeedGetter( entitiesCount++ );
            entitySeed.setCornersCount( 3 );
            entitySeed.setCornerId( 0, entity.template getSubentityIndex< 0 >( j ) );
            entitySeed.setCornerId( 1, entity.template getSubentityIndex< 0 >( k ) );
            entitySeed.setCornerId( 2, entityCenterIdx );
         }
         { // wrap around term
            auto & entitySeed = entitySeedGetter( entitiesCount++ );
            entitySeed.setCornersCount( 3 );
            entitySeed.setCornerId( 0, entity.template getSubentityIndex< 0 >( verticesCount - 1 ) );
            entitySeed.setCornerId( 1, entity.template getSubentityIndex< 0 >( 0 ) );
            entitySeed.setCornerId( 2, entityCenterIdx );
         }
      }
   }
};

template< typename MeshConfig >
struct PolygonDecomposer< MeshConfig, PolygonDecomposerVersion::ConnectEdgesToPoint >
{
   using Device = typename Devices::Host;
   using Topology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;

   static GlobalIndexType getExtraPointsCount( const MeshEntityType & entity )
   {
      return 0; // No extra points are added
   }

   static GlobalIndexType getEntitiesCount( const MeshEntityType & entity )
   {
      const auto pointsCount = entity.template getSubentitiesCount< 0 >();
      return pointsCount - 2; // there is a new triangle for every non-adjacent edge to the 0th point
   }

   template< typename MeshType, typename EntitySeedGetterType >
   static void decompose( MeshBuilder< MeshType > & meshBuilder,
                          EntitySeedGetterType entitySeedGetter,
                          GlobalIndexType & pointsCount,
                          GlobalIndexType & entitiesCount,
                          const MeshEntityType & entity )
   {
      // decompose polygon into triangles by connecting 0th point to each non-adjacent edge
      const auto verticesCount = entity.template getSubentitiesCount< 0 >();
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      for( LocalIndexType j = 1, k = 2; k < verticesCount; j++, k++ ) {
         auto & entitySeed = entitySeedGetter( entitiesCount++ );
         const auto v1 = entity.template getSubentityIndex< 0 >( j );
         const auto v2 = entity.template getSubentityIndex< 0 >( k );
         entitySeed.setCornersCount( 3 );
         entitySeed.setCornerId( 0, v0 );
         entitySeed.setCornerId( 1, v1 );
         entitySeed.setCornerId( 2, v2 );
      }
   }
};

} // namespace Meshes
} // namespace TNL