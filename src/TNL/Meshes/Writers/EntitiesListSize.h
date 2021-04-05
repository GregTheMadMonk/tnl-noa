#pragma once

#include <TNL/Meshes/Writers/VerticesPerEntity.h>

namespace TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh,
          int EntityDimension,
          typename EntityType = typename Mesh::template EntityType< EntityDimension > 
        >
struct EntitiesListSize
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getSize( const Mesh& mesh )
   {
      IndexType entitiesCount = mesh.template getEntitiesCount< EntityType >();
      IndexType verticesPerEntity = VerticesPerEntity< EntityType >::count;
      return entitiesCount * ( verticesPerEntity + 1 );
   }
};

template< typename Mesh,
          int EntityDimension,
          typename MeshConfig, 
          typename Device >
struct EntitiesListSize< Mesh, EntityDimension, MeshEntity< MeshConfig, Device, Topologies::Polygon > >
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getSize( const Mesh& mesh )
   {
      IndexType entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      IndexType entitiesListSize = entitiesCount;
      for(IndexType index = 0; index < entitiesCount; index++)
         entitiesListSize += mesh.template getSubentitiesCount< EntityDimension, 0 >( index );
      return entitiesListSize;
   }
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL
