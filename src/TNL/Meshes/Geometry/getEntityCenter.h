/***************************************************************************
                          getEntityCenter.h  -  description
                             -------------------
    begin                : Nov 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>

namespace TNL {
namespace Meshes {

// compatibility wrapper
template< typename Grid, int EntityDimension, typename Config >
__cuda_callable__
typename Grid::PointType
getEntityCenter( const Grid & grid, const GridEntity< Grid, EntityDimension, Config > & entity )
{
   return entity.getCenter();
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device > & mesh,
                 const MeshEntity< MeshConfig, Device, Topologies::Vertex > & entity )
{
   return entity.getPoint();
}

/*
 * Get an arithmetic mean of the entity's subvertices.
 *
 * For a simplex entity this corresponds to the centroid of the entity, but
 * note that other shapes such as general polygons have different formulas for
 * the centroid: https://en.wikipedia.org/wiki/Centroid#Centroid_of_a_polygon
 */
template< typename MeshConfig, typename Device, typename EntityTopology >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device > & mesh,
                 const MeshEntity< MeshConfig, Device, EntityTopology > & entity )
{
   using EntityType = MeshEntity< MeshConfig, Device, EntityTopology >;
   constexpr typename MeshConfig::LocalIndexType subvertices = EntityType::template SubentityTraits< 0 >::count;
   typename MeshTraits< MeshConfig >::PointType c = 0;
   for( typename MeshConfig::LocalIndexType i = 0;
        i < subvertices;
        i++ )
   {
      c += mesh.getPoint( entity.template getSubentityIndex< 0 >( i ) );
   }
   return ( 1.0 / subvertices ) * c;
}

} // namespace Meshes
} // namespace TNL
