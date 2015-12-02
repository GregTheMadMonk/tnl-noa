/***************************************************************************
                          tnlGridEntityCenterGetter.h  -  description
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


#ifndef TNLGRIDENTITYCENTERGETTER_H
#define	TNLGRIDENTITYCENTERGETTER_H

template< typename GridEntity >
class tnlGridEntityCenterGetter
{
};

/***
 * 1D grids
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1 > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType,1 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceStep().x() );
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0 > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType,1 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceStep().x() );
      }
};

/****
 * 2D grids
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2 > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, 2 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceStep().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceStep().y() );
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 1 > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType,2 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + 
               ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceStep().x(),
            grid.getOrigin().y() + 
               ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceStep().y() );
      }
};


template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 0 > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType,2 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + entity.getCoordinates().x() * grid.getSpaceStep().x(),
            grid.getOrigin().y() + entity.getCoordinates().y() * grid.getSpaceStep().y() );
      }
};


/***
 * 3D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, EntityDimensions > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, 3 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + 
               ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceStep().x(),
            grid.getOrigin().y() + 
               ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceStep().y(),
            grid.getOrigin().z() + 
               ( entity.getCoordinates().z() + 0.5 * entity.getBasis().z() ) * grid.getSpaceStep().z() );
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 3 > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, 3 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceStep().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceStep().y(),
            grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 ) * grid.getSpaceStep().z() );
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityCenterGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 0 > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, 3 > GridEntityType;
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      static VertexType getEntityCenter( const GridEntityType& entity )
      {
         const GridType& grid = entity.grid;
         return VertexType(
            grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceStep().x(),
            grid.getOrigin().y() + ( entity.getCoordinates().y() ) * grid.getSpaceStep().y(),
            grid.getOrigin().z() + ( entity.getCoordinates().z() ) * grid.getSpaceStep().z() );
      }
};

#endif	/* TNLGRIDENTITYCENTERGETTER_H */

