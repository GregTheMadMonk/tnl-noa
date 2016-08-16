/***************************************************************************
                          tnlOndeSidedDiffOperatorQ_impl.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps*eps;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,          
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_x = ( u[ neighbourEntities.template getEntityIndex< 1 >() ] - u[ cellIndex ] ) *
                         mesh.template getSpaceStepsProducts< -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x );          
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType& u_x_f = ( u[ neighbourEntities.template getEntityIndex< 1 >() ] - u_c ) * 
                           mesh.template getSpaceStepsProducts< -1 >();
   const RealType& u_x_b = ( u_c - u[ neighbourEntities.template getEntityIndex< -1 >() ] ) * 
                           mesh.template getSpaceStepsProducts< -1 >();   
   return ::sqrt( this->epsSquare + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b ) );
}

/***
 * 2D
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps*eps;
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,          
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType u_x = ( u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y = ( u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y ); 
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType u_x_f = ( u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y_f = ( u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, -1 >();
   const RealType u_x_b = ( u_c - u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y_b = ( u_c - u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, -1 >();
   
   return ::sqrt( this->epsSquare + 
                0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                        u_y_f * u_y_f + u_y_b * u_y_b ) );
}
/***
 * 3D
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps * eps;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,            
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c =u[ cellIndex ];
   
   const RealType u_x = ( u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y = ( u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z = ( u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, 0, -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z ); 
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   
   const RealType u_x_f = ( u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y_f = ( u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z_f = ( u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, 0, -1 >();   
   const RealType u_x_b = ( u_c - u[ neighbourEntities.template getEntityIndex< -1, 0, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y_b = ( u_c - u[ neighbourEntities.template getEntityIndex< 0, -1, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z_b = ( u_c - u[ neighbourEntities.template getEntityIndex< 0, 0, -1 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, 0, -1 >();
   
   return ::sqrt( this->epsSquare + 
                0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                        u_y_f * u_y_f + u_y_b * u_y_b + 
                        u_z_f * u_z_f + u_z_b * u_z_b ) );
}   

} // namespace Operators
} // namespace TNL