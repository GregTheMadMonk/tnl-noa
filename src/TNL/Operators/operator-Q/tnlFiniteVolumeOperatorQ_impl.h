/***************************************************************************
                          tnlFiniteVolumeOperatorQ_impl.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {   

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >   
__cuda_callable__
void  
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector, int AxeX, int AxeY, int AxeZ >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( const MeshEntity& entity,
                    const Vector& u,
                    const Real& time,
                    const IndexType& dx, 
                    const IndexType& dy,
                    const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
operator()( 
   const MeshType& mesh,
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector, int AxeX, int AxeY, int AxeZ >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
boundaryDerivative( 
   const MeshType& mesh,
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
operator()( 
   const MeshType& mesh,
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlFiniteVolumeOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    return 0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__cuda_callable__
void 
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
    CoordinatesType dimensions = mesh.getDimensions();
    CoordinatesType coordinates;
    
    for( coordinates.x()=1; coordinates.x() < dimensions.x()-1; coordinates.x()++ )
        for( coordinates.y()=1; coordinates.y() < dimensions.y()-1; coordinates.y()++  )
        {
            q.setElement( mesh.getCellIndex(coordinates), operator()( mesh, mesh.getCellIndex(coordinates), coordinates, u, time ) ); 
        }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector, int AxeX, int AxeY, int AxeZ >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( 
   const MeshType& mesh,
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   const IndexType& cellIndex = entity.getIndex();
    if ( ( AxeX == 1 ) && ( AxeY == 0 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0 >() * ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] - u[ cellIndex ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0 >() * ( u[ cellIndex ] - u[ neighbourEntities.template getEntityIndex< -1,0 >() ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,1 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,1 >() ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,-1 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,-1 >() ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 1 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1 >() * ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1 >() * ( u[ cellIndex ] - u[ neighbourEntities.template getEntityIndex< 0,-1 >() ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 1,-1 >() ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< -1,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,-1 >() ] );
    }
    return 0.0;
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
operator()( const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const IndexType& cellIndex = entity.getIndex();
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] - u[ cellIndex ] ) * 
                ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] - u[ cellIndex ] )
                * mesh.template getSpaceStepsProducts< 0, -1 >() * mesh.template getSpaceStepsProducts< 0, -1 >() + ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] - u[ cellIndex ] ) 
                * ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0 >() * mesh.template getSpaceStepsProducts< -1, 0 >() );
    if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 1, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 1, 0 ) );
    if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, -1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, -1, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, -1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, -1, 0 ) );
    if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 0, 1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 0, 1 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 0, 1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 0, 1 ) );
    if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 0, -1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0 >( mesh, entity, u, time, 0, -1 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 0, -1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1 >( mesh, entity, u, time, 0, -1 ) );
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
operator()( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
   return q.getElement( entity.getIndex() );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return String( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    this->u.bind(u);
    if(q.setSize(u.getSize()))
        return 1;
    q.setValue(0);
    return 0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__cuda_callable__
void 
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
    CoordinatesType dimensions = mesh.getDimensions();
    CoordinatesType coordinates;
    
    for( coordinates.x()=1; coordinates.x() < dimensions.x()-1; coordinates.x()++ )
        for( coordinates.y()=1; coordinates.y() < dimensions.y()-1; coordinates.y()++ )
            for( coordinates.z()=1; coordinates.z() < dimensions.z()-1; coordinates.z()++ )
                q.setElement( mesh.getCellIndex(coordinates), operator()( mesh, mesh.getCellIndex(coordinates), coordinates, u, time ) ); 
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector, int AxeX, int AxeY, int AxeZ >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( 
   const MeshType& mesh,
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const IndexType& cellIndex = entity.getIndex();    
    if ( ( AxeX == 1 ) && ( AxeY == 0 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * ( u[ cellIndex ] - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,1,0 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,1,0 >() ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,-1,0 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,-1,0 >() ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,0,1 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,0,1 >() ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.template getSpaceStepsProducts< -1, 0, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,0,-1 >() ] - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,0,-1 >() ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 1 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * ( u[ cellIndex ] - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,1,0 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 1,-1,0 >() ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< -1,1,0 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,-1,0 >() ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 0,1,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 0,-1,1 >() ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.template getSpaceStepsProducts< 0, -1, 0 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 0,1,-1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 0,-1,-1 >() ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 0 ) && ( AxeZ == 1 ) )
    {
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * ( u[ cellIndex ] - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 1,0,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 1,0,-1 >() ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< -1,0,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< -1,0,-1 >() ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 0,1,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 0,1,-1 >() ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.template getSpaceStepsProducts< 0, 0, -1 >() * 0.25 * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] + 
                   u[ neighbourEntities.template getEntityIndex< 0,-1,1 >() ] - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] -
                   u[ neighbourEntities.template getEntityIndex< 0,-1,-1 >() ] );
    }
    return 0.0;
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlFiniteVolumeOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
operator()( 
   const MeshEntity& entity,
   const Vector& u,
   const Real& time,
   const IndexType& dx, 
   const IndexType& dy,
   const IndexType& dz ) const
{
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities(); 
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const IndexType& cellIndex = entity.getIndex();     
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] ) * 
                ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] )
                * mesh.template getSpaceStepsProducts< 0, -1, 0 >() * mesh.template getSpaceStepsProducts< 0, -1, 0 >() + ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) 
                * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() * mesh.template getSpaceStepsProducts< -1, 0, 0 >()
                + ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] ) 
                * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, 0, -1 >() * mesh.template getSpaceStepsProducts< 0, 0, -1 >() );
    if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 1, 0, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 1, 0, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 1, 0, 0 ) );
    if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, -1, 0, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, -1, 0, 0 ) +
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, -1, 0, 0 ) );
    if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 1, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 1, 0 ) +
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 1, 0 ));
    if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, -1, 0 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, -1, 0 ) +
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, -1, 0 ) );
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 0, 1 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 0, 1 ) +
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 0, 1 ));
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
        return ::sqrt( this->eps + this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,1,0,0 >( mesh, entity, u, time, 0, 0, -1 ) + 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,1,0 >( mesh, entity, u, time, 0, 0, -1 ) +
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< MeshEntity, Vector,0,0,1 >( mesh, entity, u, time, 0, 0, -1 ) );
    return 0.0;
}

} // namespace Operators
} // namespace TNL
