/***************************************************************************
                          tnlTraversal_Grid3D_impl.h  -  description
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

#ifndef TNLTRAVERSAL_GRID3D_IMPL_H_
#define TNLTRAVERSAL_GRID3D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlHost, Index >, 3 >::
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
   const IndexType& zSize = grid.getDimensions().z();

   /****
    * Boundary conditions
    */
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.z() = zSize - 1;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.y() = ySize - 1;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.x() = xSize - 1;
         boundaryEntitiesProcessor.template processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }


#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.z() = 1; coordinates.z() < zSize - 1; coordinates.z() ++ )
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
tnlTraversal< tnlGrid< 3, Real, tnlHost, Index >, 2 >::
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
tnlTraversal< tnlGrid< 3, Real, tnlHost, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing edges
    */
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlHost, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing vertices
    */
}


/****
 * CUDA travelsal
 */
template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 3 >::
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
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 2 >::
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
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing edges
    */
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing vertices
    */
}



#endif /* TNLTRAVERSAL_GRID3D_IMPL_H_ */
