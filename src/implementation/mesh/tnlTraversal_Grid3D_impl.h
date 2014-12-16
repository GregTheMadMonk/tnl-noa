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
                 UserData& userData ) const
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
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.z() = zSize - 1;
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.y() = ySize - 1;
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.x() = xSize - 1;
         BoundaryEntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }


#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.z() = 1; coordinates.z() < zSize - 1; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize - 1; coordinates.x() ++ )
         {
            const IndexType index = grid.getCellIndex( coordinates );
            InteriorEntitiesProcessor::processCell( grid, userData, index, coordinates );
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
                 UserData& userData ) const
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
                 UserData& userData ) const
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
                 UserData& userData ) const
{
   /****
    * Traversing vertices
    */
}

/***
 *
 *    CUDA Specializations
 *
 */

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename UserData,
          typename BoundaryEntitiesProcessor >
__global__ void tnlTraversalGrid3DBoundaryCells( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx,
                                                 const Index gridZIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();
   const IndexType& zSize = grid->getDimensions().z();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y,
                                    ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z );

   if( cellCoordinates.x() < grid->getDimensions().x() &&
       cellCoordinates.y() < grid->getDimensions().y() &&
       cellCoordinates.z() < grid->getDimensions().z() )
   {
      if( grid->isBoundaryCell( cellCoordinates ) )
      {
         //printf( "Processing boundary conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         BoundaryEntitiesProcessor::processCell( *grid,
                                                 *userData,
                                                 grid->getCellIndex( cellCoordinates ),
                                                 cellCoordinates );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename InteriorEntitiesProcessor >
__global__ void tnlTraversalGrid3DInteriorCells( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx,
                                                 const Index gridZIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y,
                                    ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z );

   if( cellCoordinates.x() < grid->getDimensions().x() &&
       cellCoordinates.y() < grid->getDimensions().y() &&
       cellCoordinates.z() < grid->getDimensions().z())
   {
      if( ! grid->isBoundaryCell( cellCoordinates ) )
      {
         //printf( "Processing interior conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         InteriorEntitiesProcessor::processCell( *grid,
                                                 *userData,
                                                 grid->getCellIndex( cellCoordinates ),
                                                 cellCoordinates );
      }
   }
}

#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 3 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Traversing cells
    */
#ifdef HAVE_CUDA
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );

   /****
    * Boundary conditions
    */
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraversalGrid3DBoundaryCells< Real, Index, UserData, BoundaryEntitiesProcessor >
                                           <<< cudaBlocks, cudaBlockSize >>>
                                          ( kernelGrid,
                                            kernelUserData,
                                            gridXIdx,
                                            gridYIdx,
                                            gridZIdx );
         }
   cudaThreadSynchronize();
   checkCudaDevice;

   /****
    * Interior cells
    */
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraversalGrid3DInteriorCells< Real, Index, UserData, InteriorEntitiesProcessor >
                                           <<< cudaBlocks, cudaBlockSize >>>
                                          ( kernelGrid,
                                            kernelUserData,
                                            gridXIdx,
                                            gridYIdx,
                                            gridZIdx );
         }
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 3, Real, tnlCuda, Index >, 2 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
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
                 UserData& userData ) const
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
                 UserData& userData ) const
{
   /****
    * Traversing vertices
    */
}



#endif /* TNLTRAVERSAL_GRID3D_IMPL_H_ */
