#pragma once

#include <TNL/Meshes/Writers/detail/VerticesPerEntity.h>

namespace TNL {
namespace Meshes {
namespace Writers {
namespace detail {

template< typename Mesh,
          int EntityDimension,
          typename EntityType = typename Mesh::template EntityType< EntityDimension > >
struct VTKOffsetsCountGetter
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getOffsetsCount( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const IndexType verticesPerEntity = VerticesPerEntity< EntityType >::count;
      return entitiesCount * verticesPerEntity;
   }
};

template< typename Mesh,
          int EntityDimension,
          typename MeshConfig,
          typename Device >
struct VTKOffsetsCountGetter< Mesh, EntityDimension, MeshEntity< MeshConfig, Device, Topologies::Polygon > >
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType getOffsetsCount( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      IndexType offsetsCount = 0;
      for(IndexType index = 0; index < entitiesCount; index++)
         offsetsCount += mesh.template getSubentitiesCount< EntityDimension, 0 >( index );
      return offsetsCount;
   }
};

} // namespace detail
} // namespace Writers
} // namespace Meshes
} // namespace TNL
