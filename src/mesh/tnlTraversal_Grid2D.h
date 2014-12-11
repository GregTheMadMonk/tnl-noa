/***************************************************************************
                          tnlTraversal_Grid2D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
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


#ifndef TNLTRAVERSAL_GRID2D_H_
#define TNLTRAVERSAL_GRID2D_H_


template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 2 >
{
   public:
      typedef tnlGrid< 2, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;
};

template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 1 >
{
   public:
      typedef tnlGrid< 2, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;
};

template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 0 >
{
   public:
      typedef tnlGrid< 2, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;
};


template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 2 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;

};

template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 1 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;

};

template< typename Real,
          typename Index >
class tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 0 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename BoundaryEntitiesProcessor,
                typename InteriorEntitiesProcessor >
      void processEntities( const GridType& grid,
                            UserData& userData ) const;

};


#include <implementation/mesh/tnlTraversal_Grid2D_impl.h>

#endif /* TNLTRAVERSAL_GRID2D_H_ */
