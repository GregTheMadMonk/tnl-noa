#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>
#include <functional>

namespace TNL {
namespace Meshes {

enum class EntityDecomposerVersion
{
   ConnectEdgesToCentroid, ConnectEdgesToPoint
};

template< typename MeshConfig,
          typename Topology,
          EntityDecomposerVersion EntityDecomposerVersion_ = EntityDecomposerVersion::ConnectEdgesToCentroid, 
          EntityDecomposerVersion SubentityDecomposerVersion = EntityDecomposerVersion::ConnectEdgesToCentroid >
struct EntityDecomposer;

// Polygon
template< typename MeshConfig, EntityDecomposerVersion SubentityDecomposerVersion > // SubentityDecomposerVersion is not used for polygons
struct EntityDecomposer< MeshConfig, Topologies::Polygon, EntityDecomposerVersion::ConnectEdgesToCentroid, SubentityDecomposerVersion >
{
   using Device = typename Devices::Host;
   using Topology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using VertexMeshEntityType = typename MeshEntityType::template SubentityTraits< 0 >::SubentityType;
   using PointType = typename VertexMeshEntityType::PointType;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;
   using PointCreationFunctorType = std::function< GlobalIndexType ( const PointType& ) >;
   using DecomposedEntityFunctorType = std::function< void ( GlobalIndexType, GlobalIndexType, GlobalIndexType ) >;
   
   static std::pair< GlobalIndexType, GlobalIndexType > getExtraPointsAndEntitiesCount( const MeshEntityType & entity )
   {
      const auto pointsCount = entity.template getSubentitiesCount< 0 >();
      if( pointsCount == 3 ) // polygon is triangle
         return { 0, 1 }; // No extra points and no decomposition
      return { 1, pointsCount }; // 1 extra centroid point and decomposition creates pointsCount triangles
   }

   static void decompose( const MeshEntityType & entity,
                          PointCreationFunctorType pointCreationFunctor,
                          DecomposedEntityFunctorType decomposedEntityFunctor )
   {
      const auto verticesCount = entity.template getSubentitiesCount< 0 >();
      if( verticesCount == 3 ) { // polygon is only copied as it's already a triangle
         const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
         const auto v1 = entity.template getSubentityIndex< 0 >( 1 );
         const auto v2 = entity.template getSubentityIndex< 0 >( 2 );
         decomposedEntityFunctor( v0, v1, v2 );
      }
      else { // polygon is decomposed as it has got more than 3 vertices
         const auto v0 = pointCreationFunctor( getEntityCenter( entity.getMesh(), entity ) );
         // decompose polygon into triangles by connecting each edge to the centroid
         for( LocalIndexType j = 0, k = 1; k < verticesCount; j++, k++ ) {
            const auto v1 = entity.template getSubentityIndex< 0 >( j );
            const auto v2 = entity.template getSubentityIndex< 0 >( k );
            decomposedEntityFunctor( v0, v1, v2 );
         }
         { // wrap around term
            const auto v1 = entity.template getSubentityIndex< 0 >( verticesCount - 1 );
            const auto v2 = entity.template getSubentityIndex< 0 >( 0 );
            decomposedEntityFunctor( v0, v1, v2 );
         }
      }
   }
};

template< typename MeshConfig, EntityDecomposerVersion SubentityDecomposerVersion > // SubentityDecomposerVersion is not used for polygons
struct EntityDecomposer< MeshConfig, Topologies::Polygon, EntityDecomposerVersion::ConnectEdgesToPoint, SubentityDecomposerVersion >
{
   using Device = typename Devices::Host;
   using Topology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using VertexMeshEntityType = typename MeshEntityType::template SubentityTraits< 0 >::SubentityType;
   using PointType = typename VertexMeshEntityType::PointType;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;
   using PointCreationFunctorType = std::function< GlobalIndexType ( const PointType& ) >;
   using DecomposedEntityFunctorType = std::function< void ( GlobalIndexType, GlobalIndexType, GlobalIndexType ) >;

   static std::pair< GlobalIndexType, GlobalIndexType > getExtraPointsAndEntitiesCount( const MeshEntityType & entity )
   {
      const auto pointsCount = entity.template getSubentitiesCount< 0 >();
      return { 0, pointsCount - 2 }; // no extra points and there is a triangle for every non-adjacent edge to the 0th point (pointsCount - 2)
   }

   static void decompose( const MeshEntityType & entity,
                          PointCreationFunctorType pointCreationFunctor,
                          DecomposedEntityFunctorType decomposedEntityFunctor )
   {
      // decompose polygon into triangles by connecting 0th point to each non-adjacent edge
      const auto verticesCount = entity.template getSubentitiesCount< 0 >();
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      for( LocalIndexType j = 1, k = 2; k < verticesCount; j++, k++ ) {
         const auto v1 = entity.template getSubentityIndex< 0 >( j );
         const auto v2 = entity.template getSubentityIndex< 0 >( k );
         decomposedEntityFunctor( v0, v1, v2 );
      }
   }
};

// Polyhedron
template< typename MeshConfig, EntityDecomposerVersion SubentityDecomposerVersion >
struct EntityDecomposer< MeshConfig, Topologies::Polyhedron, EntityDecomposerVersion::ConnectEdgesToCentroid, SubentityDecomposerVersion >
{
   using Device = typename Devices::Host;
   using CellTopology = typename Topologies::Polyhedron;
   using FaceTopology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, CellTopology >;
   using VertexMeshEntityType = typename MeshEntityType::template SubentityTraits< 0 >::SubentityType;
   using PointType = typename VertexMeshEntityType::PointType;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;
   using PointCreationFunctorType = std::function< GlobalIndexType ( const PointType& ) >;
   using DecomposedEntityFunctorType = std::function< void ( GlobalIndexType, GlobalIndexType, GlobalIndexType, GlobalIndexType ) >;
   using SubentityDecomposer = EntityDecomposer< MeshConfig, FaceTopology, SubentityDecomposerVersion >;
   
   static std::pair< GlobalIndexType, GlobalIndexType > getExtraPointsAndEntitiesCount( const MeshEntityType & entity )
   {
      const auto& mesh = entity.getMesh();
      GlobalIndexType extraPointsCount = 1, // there is one new centroid point
                      entitiesCount = 0;
      const auto facesCount = entity.template getSubentitiesCount< 2 >();
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( i ) );

         GlobalIndexType faceExtraPoints, faceEntitiesCount;
         std::tie( faceExtraPoints, faceEntitiesCount ) = SubentityDecomposer::getExtraPointsAndEntitiesCount( face );
         extraPointsCount += faceExtraPoints; // add extra points from decomposition of faces
         entitiesCount += faceEntitiesCount; // there is a new tetrahedron per triangle of a face
      }
      return { extraPointsCount, entitiesCount };
   }

   static void decompose( const MeshEntityType & entity,
                          PointCreationFunctorType pointCreationFunctor,
                          DecomposedEntityFunctorType decomposedEntityFunctor )
   {
      const auto & mesh = entity.getMesh();
      const auto v3 = pointCreationFunctor( getEntityCenter( mesh, entity ) );

      // Lambda for creating tetrahedron from decomposed triangles of faces
      auto decomposedSubentityFunctor = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
         decomposedEntityFunctor( v0, v1, v2, v3 );
      };

      const auto facesCount = entity.template getSubentitiesCount< 2 >();
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( i ) );
         SubentityDecomposer::decompose( face, pointCreationFunctor, decomposedSubentityFunctor );
      }
   }
};

template< typename MeshConfig, EntityDecomposerVersion SubentityDecomposerVersion >
struct EntityDecomposer< MeshConfig, Topologies::Polyhedron, EntityDecomposerVersion::ConnectEdgesToPoint, SubentityDecomposerVersion >
{
   // https://mathoverflow.net/a/7672
   using Device = typename Devices::Host;
   using CellTopology = typename Topologies::Polyhedron;
   using FaceTopology = typename Topologies::Polygon;
   using MeshEntityType = MeshEntity< MeshConfig, Device, CellTopology >;
   using MeshSubentityType = MeshEntity< MeshConfig, Device, FaceTopology >;
   using VertexMeshEntityType = typename MeshEntityType::template SubentityTraits< 0 >::SubentityType;
   using PointType = typename VertexMeshEntityType::PointType;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;
   using RealType = typename MeshConfig::RealType;
   using PointCreationFunctorType = std::function< GlobalIndexType ( const PointType& ) >;
   using DecomposedEntityFunctorType = std::function< void ( GlobalIndexType, GlobalIndexType, GlobalIndexType, GlobalIndexType ) >;
   using SubentityDecomposer = EntityDecomposer< MeshConfig, FaceTopology, SubentityDecomposerVersion >;
   
   static std::pair< GlobalIndexType, GlobalIndexType > getExtraPointsAndEntitiesCount( const MeshEntityType & entity )
   {
      const auto& mesh = entity.getMesh();
      const auto v3 = entity.template getSubentityIndex< 0 >( 0 );
      GlobalIndexType extraPointsCount = 0,
                      entitiesCount = 0;
      const auto facesCount = entity.template getSubentitiesCount< 2 >();
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( i ) );
         if( !faceContainsPoint( face, v3 ) ) { // include only faces, that don't contain point v3
            GlobalIndexType faceExtraPoints, faceEntitiesCount;
            std::tie( faceExtraPoints, faceEntitiesCount ) = SubentityDecomposer::getExtraPointsAndEntitiesCount( face );
            extraPointsCount += faceExtraPoints; // add extra points from decomposition of faces
            entitiesCount += faceEntitiesCount; // there is a new tetrahedron per triangle of a face
         }
      }
      return { extraPointsCount, entitiesCount };
   }

   static void decompose( const MeshEntityType & entity,
                          PointCreationFunctorType pointCreationFunctor,
                          DecomposedEntityFunctorType decomposedEntityFunctor )
   {
      const auto & mesh = entity.getMesh();
      const auto v3 = entity.template getSubentityIndex< 0 >( 0 );

      // Lambda for creating tetrahedron by connecting decomposed triangles of faces to 0th point of polyhedron
      auto decomposedSubentityFunctor = [&] ( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 ) {
         decomposedEntityFunctor( v0, v1, v2, v3 );
      };

      const auto facesCount = entity.template getSubentitiesCount< 2 >();
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( i ) );
         if( !faceContainsPoint( face, v3 ) ) { // include only faces, that don't contain point v3
            SubentityDecomposer::decompose( face, pointCreationFunctor, decomposedSubentityFunctor );
         }
      }
   }

private:
   static bool faceContainsPoint( const MeshSubentityType & face, const GlobalIndexType point )
   {
      const LocalIndexType pointsCount = face.template getSubentitiesCount< 0 >();
      for( LocalIndexType i = 0; i < pointsCount; i++ ) {
         const auto facePoint = face.template getSubentityIndex< 0 >( i );
         if( point == facePoint )
            return true;
      }
      return false;
   }
};

} // namespace Meshes
} // namespace TNL