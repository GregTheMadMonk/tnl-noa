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
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Boundary cells
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::processCell( grid, userData, 0, coordinates );
   coordinates.x() = xSize - 1;
   EntitiesProcessor::processCell( grid, userData, xSize - 1, coordinates );
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize-1; coordinates.x() ++ )
   {
      const IndexType index = grid.getCellIndex( coordinates );
      EntitiesProcessor::processCell( grid, userData, index, coordinates );
   }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::processVertex( grid, userData, 0, coordinates );
   coordinates.x() = xSize;
   EntitiesProcessor::processVertex( grid, userData, xSize - 1, coordinates );
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
   {
      const IndexType index = grid.getVertexIndex( coordinates );
      EntitiesProcessor::processVertex( grid, userData, index, coordinates );
   }
}

/*****
 *
 *  CUDA specialization
 *
 */

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor >
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
         printf( "Processing boundary conditions at %d \n", cellCoordinates.x() );
         EntitiesProcessor::processCell( *grid,
                                         *userData,
                                         grid->getCellIndex( cellCoordinates ),
                                         cellCoordinates );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor >
__global__ void tnlTraversalGrid1DInteriorCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   CoordinatesType cellCoordinates;
   cellCoordinates.x() = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( cellCoordinates.x() > 0 && cellCoordinates.x() < grid->getDimensions().x() - 1 )
   {
      printf( "Processing interior cell at %d \n", cellCoordinates.x() );
      const IndexType index = grid->getCellIndex( cellCoordinates );
      EntitiesProcessor::processCell( *grid, *userData, index, cellCoordinates );
   }
}

#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA

   /****
    * Boundary conditions
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      tnlTraversalGrid1DBoundaryCells< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
   cudaThreadSynchronize();
   checkCudaDevice;
#endif
}
template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Interior cells
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );


   cerr << "Processing interior cells ..................." << endl;
   dim3 cudaGridSize;
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx++ )
   {
      if( gridXIdx == cudaXGrids - 1 )
         cudaGridSize.x = cudaBlocks.x % tnlCuda::getMaxGridSize();
      tnlTraversalGrid1DInteriorCells< Real, Index, UserData, EntitiesProcessor >
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
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing vertices
    */

}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing vertices
    */

}


#endif /* TNLTRAVERSAL_GRID1D_IMPL_H_ */
