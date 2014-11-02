/***************************************************************************
                          tnlTraversal_Grid2D_impl.h  -  description
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


#ifndef TNLTRAVERSAL_GRID2D_IMPL_H_
#define TNLTRAVERSAL_GRID2D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 2 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitiesProcessor ) const
{
   /****
    * Traversing cells
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

   /****
    * Boundary conditions
    */
   for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
   {
      coordinates.y() = 0;
      boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      coordinates.y() = ySize - 1;
      boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
   }
   for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
   {
      coordinates.x() = 0;
      boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      coordinates.x() = xSize - 1;
      boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
   }

   /****
    * Interior cells
    */
#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
      for( coordinates.x() = 1; coordinates.x() < xSize - 1; coordinates.x() ++ )
      {
         const IndexType index = grid.getCellIndex( coordinates );
         interiorEntitiesProcessor.template processCell( grid, userData, index, coordinates );
      }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing faces
    */
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlHost, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing vertices
    */
}


template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 2 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing cells
    */
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing faces
    */
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 2, Real, tnlCuda, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing vertices
    */
}


#endif /* TNLTRAVERSAL_GRID2D_IMPL_H_ */
