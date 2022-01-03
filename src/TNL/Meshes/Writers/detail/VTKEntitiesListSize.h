#pragma once

#include <TNL/Meshes/Writers/detail/VerticesPerEntity.h>

namespace TNL {
namespace Meshes {
namespace Writers {
namespace detail {

template< typename Mesh,
          int EntityDimension,
          typename EntityType = typename Mesh::template EntityType< EntityDimension > >
struct VTKEntitiesListSize
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getSize( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const IndexType verticesPerEntity = VerticesPerEntity< EntityType >::count;
      return entitiesCount * ( verticesPerEntity + 1 );
   }
};

template< typename Mesh,
          int EntityDimension,
          typename MeshConfig,
          typename Device >
struct VTKEntitiesListSize< Mesh, EntityDimension, MeshEntity< MeshConfig, Device, Topologies::Polygon > >
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getSize( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      IndexType entitiesListSize = entitiesCount;
      for(IndexType index = 0; index < entitiesCount; index++)
         entitiesListSize += mesh.template getSubentitiesCount< EntityDimension, 0 >( index );
      return entitiesListSize;
   }
};

} // namespace detail
} // namespace Writers
} // namespace Meshes
} // namespace TNL
