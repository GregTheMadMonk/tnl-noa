/***************************************************************************
                          BoundaryGridEntityChecker.h  -  description
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
class BoundaryGridEntityChecker
{
};

/***
 * 1D grids
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config  > >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 1, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() - 1 );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 1, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() );
      }
};

/****
 * 2D grids
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 2, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() - 1 ||
                 entity.getCoordinates().y() == entity.grid.getDimensions().y() - 1 );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 2, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() ==
                    entity.grid.getDimensions().x() - entity.getBasis().x() ||
                 entity.getCoordinates().y() ==
                    entity.grid.getDimensions().y() - entity.getBasis().y() );
 
      }
};


template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 2, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() ||
                 entity.getCoordinates().y() == entity.grid.getDimensions().y() );
 
      }
};


/***
 * 3D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config ,
          int EntityDimensions >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, EntityDimensions, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 3, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().z() == 0 ||
                 entity.getCoordinates().x() ==
                    entity.grid.getDimensions().x() - entity.getBasis().x() ||
                 entity.getCoordinates().y() ==
                    entity.grid.getDimensions().y() - entity.getBasis().y() ||
                 entity.getCoordinates().z() ==
                    entity.grid.getDimensions().z() - entity.getBasis().z() );
 
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 3, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().z() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() - 1 ||
                 entity.getCoordinates().y() == entity.grid.getDimensions().y() - 1 ||
                 entity.getCoordinates().z() == entity.grid.getDimensions().z() - 1 );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0, Config > >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, 3, Config > GridEntityType;
 
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().z() == 0 ||
                 entity.getCoordinates().x() == entity.grid.getDimensions().x() ||
                 entity.getCoordinates().y() == entity.grid.getDimensions().y() ||
                 entity.getCoordinates().z() == entity.grid.getDimensions().z() );
      }
};

} // namespace Meshes
} // namespace TNL

