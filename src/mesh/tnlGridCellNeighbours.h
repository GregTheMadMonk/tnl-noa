/***************************************************************************
                          tnlGridCellNeighbours.h  -  description
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

#ifndef TNLGRIDCELLNEIGHBOURS_H_
#define TNLGRIDCELLNEIGHBOURS_H_

#include <mesh/tnlGrid.h>

template< typename Grid >
class tnlGridCellNeighbours{};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridCellNeighbours< tnlGrid< 1, Real, Device, Index > >
{
   public:
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      enum { Dimensions = GridType::Dimensions };

      tnlGridCellNeighbours( const GridType& grid,
                             const CoordinatesType& cellCoordinates );

      const IndexType& getXPredecessor() const;

      const IndexType& getXSuccessor() const;

   protected:
      IndexType xPredecessor, xSuccessor;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      enum { Dimensions = GridType::Dimensions };

      tnlGridCellNeighbours( const GridType& grid,
                             const CoordinatesType& cellCoordinates );

      const IndexType& getXPredecessor() const;

      const IndexType& getXSuccessor() const;

      const IndexType& getYPredecessor() const;

      const IndexType& getYSuccessor() const;


   protected:
      IndexType xPredecessor, xSuccessor,
                yPredecessor, ySuccessor;
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      enum { Dimensions = GridType::Dimensions };

      tnlGridCellNeighbours( const GridType& grid,
                             const CoordinatesType& cellCoordinates );

      const IndexType& getXPredecessor() const;

      const IndexType& getXSuccessor() const;

      const IndexType& getYPredecessor() const;

      const IndexType& getYSuccessor() const;

      const IndexType& getZPredecessor() const;

      const IndexType& getZSuccessor() const;

   protected:
      IndexType xPredecessor, xSuccessor,
                yPredecessor, ySuccessor,
                zPredecessor, zSuccessor;
};

#include <implementation/mesh/tnlGridCellNeighbours_impl.h>

#endif /* TNLGRIDCELLNEIGHBOURS_H_ */
