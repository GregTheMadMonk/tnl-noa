/***************************************************************************
                          tnlGridCellNeighbours_impl.h  -  description
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

#ifndef TNLGRIDCELLNEIGHBOURS_IMPL_H_
#define TNLGRIDCELLNEIGHBOURS_IMPL_H_

template< typename Real,
          typename Device,
          typename Index >
tnlGridCellNeighbours< tnlGrid< 1, Real, Device, Index > >::
tnlGridCellNeighbours( const GridType& grid,
                       const CoordinatesType& cellCoordinates )
{
   const IndexType cellIndex = grid.getCellIndex( cellCoordinates );
   this->xPredecessor = cellIndex - 1;
   this->xSuccessor = cellIndex + 1;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 1, Real, Device, Index > >::
getXPredecessor() const
{
   return this->xPredecessor;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 1, Real, Device, Index > >::
getXSuccessor() const
{
   return this->xSuccessor;
}

template< typename Real,
          typename Device,
          typename Index >
tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >::
tnlGridCellNeighbours( const GridType& grid,
                       const CoordinatesType& cellCoordinates )
{
   const IndexType cellIndex = grid.getCellIndex( cellCoordinates );
   this->xPredecessor = cellIndex - 1;
   this->xSuccessor = cellIndex + 1;
   this->yPredecessor = cellIndex - grid.getDimensions().x();
   this->ySuccessor = cellIndex + grid.getDimensions().x();
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >::
getXPredecessor() const
{
   return this->xPredecessor;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >::
getXSuccessor() const
{
   return this->xSuccessor;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >::
getYPredecessor() const
{
   return this->yPredecessor;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlGridCellNeighbours< tnlGrid< 2, Real, Device, Index > >::
getYSuccessor() const
{
   return this->ySuccessor;
}




#endif /* TNLGRIDCELLNEIGHBOURS_IMPL_H_ */
