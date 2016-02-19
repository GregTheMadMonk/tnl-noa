/***************************************************************************
                          tnlTraverser_Grid1D.h  -  description
                             -------------------
    begin                : Jul 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLTRAVERSER_GRID1D_H_
#define TNLTRAVERSER_GRID1D_H_

#include <mesh/tnlTraverser.h>

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 1, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
      
};


template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 1, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
};

#ifdef UNDEF

/****
 * CUDA traversals
 */
template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;


      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
      
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
      


};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;


      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
      
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;      
};

#endif

#include <mesh/grids/tnlTraverser_Grid1D_impl.h>

#endif /* TNLTRAVERSER_GRID1D_H_ */
