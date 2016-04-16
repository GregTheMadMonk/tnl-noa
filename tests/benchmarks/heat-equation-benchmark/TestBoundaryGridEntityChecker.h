/***************************************************************************
                          tnlBoundaryGridEntityChecker.h  -  description
                             -------------------
    begin                : Dec 2, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

template< typename GridEntity >
class TestBoundaryGridEntityChecker
{
};

/***
 * 1D grids
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 1, Real, Device, Index >, 1, Config  > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 1, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 1, Real, Device, Index >, 0, Config > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 1, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 2, Real, Device, Index >, 2, Config > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 2, Config > GridEntityType;
      
      __cuda_callable__ inline
      static bool isBoundaryEntity( const GridEntityType& entity )
      {
         return( entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 ||
                 entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename Config  >
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 2, Real, Device, Index >, 1, Config > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 2, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 2, Real, Device, Index >, 0, Config > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 2, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 3, Real, Device, Index >, EntityDimensions, Config > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 3, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 3, Real, Device, Index >, 3, Config > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 3, Config > GridEntityType;
      
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
class TestBoundaryGridEntityChecker< TestGridEntity< tnlGrid< 3, Real, Device, Index >, 0, Config > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef TestGridEntity< GridType, 3, Config > GridEntityType;
      
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


