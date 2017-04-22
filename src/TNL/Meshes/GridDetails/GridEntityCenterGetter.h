/***************************************************************************
                          GridEntityCenterGetter.h  -  description
                             -------------------
    begin                : Dec 2, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {
namespace Meshes {

template< typename GridEntity >
class GridEntityCenterGetter
{
};

/***
 * 1D grids
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config > >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 1, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x() );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 0, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x() );
      }
};

/****
 * 2D grids
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 2, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y() );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 1, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() +
               ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() +
               ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceSteps().y() );
      }
};


template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 0, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + entity.getCoordinates().x() * grid.getSpaceSteps().x(),
            grid.getOrigin().y() + entity.getCoordinates().y() * grid.getSpaceSteps().y() );
      }
};


/***
 * 3D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, EntityDimension, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() +
               ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() +
               ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceSteps().y(),
            grid.getOrigin().z() +
               ( entity.getCoordinates().z() + 0.5 * entity.getBasis().z() ) * grid.getSpaceSteps().z() );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 3, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y(),
            grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 ) * grid.getSpaceSteps().z() );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 0, Config > GridEntityType;
      typedef typename GridType::VertexType VertexType;
 
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.getMesh();
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() ) * grid.getSpaceSteps().y(),
            grid.getOrigin().z() + ( entity.getCoordinates().z() ) * grid.getSpaceSteps().z() );
      }
};

} // namespace Meshes
} // namespace TNL

