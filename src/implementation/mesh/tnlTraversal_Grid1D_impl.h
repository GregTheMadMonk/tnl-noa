/***************************************************************************
                          tnlTraversal_Grid1D_impl.h  -  description
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

#ifndef TNLTRAVERSAL_GRID1D_IMPL_H_
#define TNLTRAVERSAL_GRID1D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Traversing cells
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();

   /****
    * Boundary conditions
    */
   coordinates.x() = 0;
   BoundaryEntitiesProcessor::processCell( grid, userData, 0, coordinates );
   coordinates.x() = xSize - 1;
   BoundaryEntitiesProcessor::processCell( grid, userData, xSize - 1, coordinates );

   /****
    * Interior cells
    */
   for( coordinates.x() = 1; coordinates.x() < xSize-1; coordinates.x() ++ )
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
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Traversing vertices
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
   {
      const IndexType index = grid.getVertexIndex( coordinates );
      if( grid.isBoundaryVertex( coordinates ) )
         BoundaryEntitiesProcessor::processVertices( grid, userData, index, coordinates );
      else
         InteriorEntitiesProcessor::processVertices( grid, userData, index, coordinates );
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename UserData,
          typename BoundaryEntitiesProcessor >
__global__ void tnlTraversalGrid1DBoundaryCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 Index gridXIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x );

   if( cellCoordinates.x() < grid->getDimensions().x() )
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
__global__ void tnlTraversalGrid1DInteriorCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   CoordinatesType coordinates;
   coordinates.x() = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   InteriorEntitiesProcessor p;
   if( coordinates.x() > 0 && coordinates.x() < grid->getDimensions().x() - 1 )
   {
      const IndexType index = grid->getCellIndex( coordinates );
      InteriorEntitiesProcessor::processCell( *grid, *userData, index, coordinates );
   }
}

#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename BoundaryEntitiesProcessor,
             typename InteriorEntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 1 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Traversing cells
    */
#ifdef HAVE_CUDA
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   /****
    * Boundary conditions
    */
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      tnlTraversalGrid1DBoundaryCells< Real, Index, UserData, BoundaryEntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
   cudaThreadSynchronize();
   checkCudaDevice;

   /****
    * Interior cells
    */
   dim3 cudaGridSize;
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx++ )
   {
      if( gridXIdx == cudaXGrids - 1 )
         cudaGridSize.x = cudaBlocks.x % tnlCuda::getMaxGridSize();
      tnlTraversalGrid1DInteriorCells< Real, Index, UserData, InteriorEntitiesProcessor >
                                     <<< cudaGridSize, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
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
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 0 >::
processEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Traversing vertices
    */

}



#endif /* TNLTRAVERSAL_GRID1D_IMPL_H_ */
