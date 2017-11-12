/***************************************************************************
                          getEntityCenter.h  -  description
                             -------------------
    begin                : Nov 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/CudaCallable.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>

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

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device > & mesh,
                 const MeshEntity< MeshConfig, Device, Topologies::Edge > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    return 0.5 * ( v0.getPoint() + v1.getPoint() );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device > & mesh,
                 const MeshEntity< MeshConfig, Device, Topologies::Triangle > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
    return ( 1.0 / 3.0 ) * ( v0.getPoint() + v1.getPoint() + v2.getPoint() );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device > & mesh,
                 const MeshEntity< MeshConfig, Device, Topologies::Tetrahedron > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );
    return 0.25 * ( v0.getPoint() + v1.getPoint() + v2.getPoint() + v3.getPoint() );
}

} // namespace Meshes
} // namespace TNL
