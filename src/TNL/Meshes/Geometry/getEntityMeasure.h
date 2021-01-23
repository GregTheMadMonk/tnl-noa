/***************************************************************************
                          getEntityMeasure.h  -  description
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
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>

namespace TNL {
namespace Meshes {

template< typename Grid, typename Config >
__cuda_callable__
typename Grid::RealType
getEntityMeasure( const Grid & grid, const GridEntity< Grid, 0, Config > & entity )
{
    // entity.getMeasure() returns 0.0 !!!
    return 1.0;
}

template< typename Grid, int EntityDimension, typename Config >
__cuda_callable__
typename Grid::RealType
getEntityMeasure( const Grid & grid, const GridEntity< Grid, EntityDimension, Config > & entity )
{
    return entity.getMeasure();
}

// Vertex
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Vertex > & entity )
{
    return 1.0;
}

// Edge
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Edge > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    return l2Norm( v1 - v0 );
}

// Triangle
template< typename VectorExpression,
          std::enable_if_t< VectorExpression::getSize() == 2, bool > = true >
__cuda_callable__
typename VectorExpression::RealType
getTriangleArea( const VectorExpression & v1,
                 const VectorExpression & v2 )
{
    using Real = typename VectorExpression::RealType;
    return Real( 0.5 ) * TNL::abs( v1.x() * v2.y() - v1.y() * v2.x() );
}

template< typename VectorExpression,
          std::enable_if_t< VectorExpression::getSize() == 3, bool > = true >
__cuda_callable__
typename VectorExpression::RealType
getTriangleArea( const VectorExpression & v1,
                 const VectorExpression & v2 )
{
    using Real = typename VectorExpression::RealType;
    // formula from http://math.stackexchange.com/a/128999
    const Real c1 = v1.y() * v2.z() - v1.z() * v2.y();   // first component of the cross product
    const Real c2 = v1.z() * v2.x() - v1.x() * v2.z();   // second component of the cross product
    const Real c3 = v1.x() * v2.y() - v1.y() * v2.x();   // third component of the cross product
    return Real( 0.5 ) * TNL::sqrt( c1 * c1 + c2 * c2 + c3 * c3 );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Triangle > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    return getTriangleArea( v2 - v0, v1 - v0 );
}

// Quadrangle
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Quadrangle > & entity )
{
    // measure = 0.5 * |AC x BD|, where AC and BD are the diagonals
    // Hence, we can use the same formula as for the triangle area.
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    return getTriangleArea( v2 - v0, v3 - v1 );
}

template< typename VectorExpression >
__cuda_callable__
typename VectorExpression::RealType
getTetrahedronVolume( const VectorExpression& v1,
                      const VectorExpression& v2,
                      const VectorExpression& v3 )
{
    using Real = typename VectorExpression::RealType;
    // V = (1/6) * det(v1, v2, v3)
    const Real det = v1.x() * v2.y() * v3.z() +
                     v1.y() * v2.z() * v3.x() +
                     v1.z() * v2.x() * v3.y() -
                   ( v1.z() * v2.y() * v3.x() +
                     v1.y() * v2.x() * v3.z() +
                     v1.x() * v2.z() * v3.y() );
    return Real( 1.0 / 6.0 ) * TNL::abs( det );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Tetrahedron > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    return getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Hexahedron > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    const auto& v4 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 4 ) );
    const auto& v5 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 5 ) );
    const auto& v6 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 6 ) );
    const auto& v7 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 7 ) );
    // https://www.cfd-online.com/Forums/main/163122-volume-general-hexahedron.html#post574650
    return getTetrahedronVolume( v0 - v4, v3 - v4, v1 - v4 )
         + getTetrahedronVolume( v2 - v4, v3 - v4, v1 - v4 )
         + getTetrahedronVolume( v1 - v4, v2 - v4, v5 - v4 )
         + getTetrahedronVolume( v6 - v4, v2 - v4, v5 - v4 )
         + getTetrahedronVolume( v3 - v4, v2 - v4, v7 - v4 )
         + getTetrahedronVolume( v6 - v4, v2 - v4, v7 - v4 );
}

} // namespace Meshes
} // namespace TNL
