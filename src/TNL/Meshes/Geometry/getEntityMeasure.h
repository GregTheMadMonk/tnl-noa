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
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>

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

// TODO: move to StaticVector::norm
template< typename Real >
__cuda_callable__
Real
getVectorLength( const TNL::Containers::StaticVector< 1, Real > & vector )
{
    return TNL::abs( vector[ 0 ] );
}

template< typename VectorExpression >
__cuda_callable__
typename VectorExpression::RealType
getVectorLength( const VectorExpression& expr )
{
    return TNL::sqrt( TNL::dot( expr, expr ) );
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
    return getVectorLength( v1 - v0 );
}

// Triangle
template< typename Real >
__cuda_callable__
Real
getTriangleArea( const TNL::Containers::StaticVector< 2, Real > & v1,
                 const TNL::Containers::StaticVector< 2, Real > & v2 )
{
    return 0.5 * TNL::abs( v1.x() * v2.y() - v1.y() * v2.x() );
}

template< typename Real >
__cuda_callable__
Real
getTriangleArea( const TNL::Containers::StaticVector< 3, Real > & v1,
                 const TNL::Containers::StaticVector< 3, Real > & v2 )
{
    // formula from http://math.stackexchange.com/a/128999
    Real S = 0.0;
    Real aux = v1.y() * v2.z() - v1.z() * v2.y();   // first component of the cross product
    S += aux * aux;
    aux = v1.z() * v2.x() - v1.x() * v2.z();        // second component of the cross product
    S += aux * aux;
    aux = v1.x() * v2.y() - v1.y() * v2.x();        // third component of the cross product
    S += aux * aux;
    return 0.5 * ::sqrt( S );
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
    using Point = std::decay_t< decltype( v0 ) >;
    const Point p1 = v2 - v0;
    const Point p2 = v1 - v0;
    return getTriangleArea( p1, p2 );
}

// Quadrilateral
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Quadrilateral > & entity )
{
    // measure = 0.5 * |AC x BD|, where AC and BD are the diagonals
    // Hence, we can use the same formula as for the triangle area.
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    using Point = std::decay_t< decltype( v0 ) >;
    const Point p1 = v2 - v0;
    const Point p2 = v3 - v1;
    return getTriangleArea( p1, p2 );
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
    return ( 1.0 / 6.0 ) * TNL::abs( det );
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

} // namespace Meshes
} // namespace TNL
