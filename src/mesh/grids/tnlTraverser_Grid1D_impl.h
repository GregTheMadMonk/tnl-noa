/***************************************************************************
                          tnlTraverser_Grid1D_impl.h  -  description
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

#ifndef TNLTRAVERSER_GRID1D_IMPL_H_
#define TNLTRAVERSER_GRID1D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlHost, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                 UserData& userData ) const
{
   /****
    * Boundary cells
    */
   typedef typename Grid::Cell CellTopology;
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::template processEntity< CellTopology >( grid, userData, 0, coordinates );
   coordinates.x() = xSize - 1;
   EntitiesProcessor::template processEntity< CellTopology >( grid, userData, xSize - 1, coordinates );
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlHost, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   typedef typename Grid::Cell CellTopology;
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize-1; coordinates.x() ++ )
   {
      const IndexType index = grid.getCellIndex( coordinates );
      EntitiesProcessor::template processEntity< CellTopology >( grid, userData, index, coordinates );
   }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlHost, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   typedef typename Grid::Vertex VertexTopology;
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::template processEntity< VertexTopology >( grid, userData, 0, coordinates );
   coordinates.x() = xSize;
   EntitiesProcessor::template processEntity< VertexTopology >( grid, userData, xSize - 1, coordinates );
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlHost, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   typedef typename Grid::Vertex VertexTopology;
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
   {
      const IndexType index = grid.getVertexIndex( coordinates );
      EntitiesProcessor::template processEntity< VertexTopology >( grid, userData, index, coordinates );
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
__global__ void tnlTraverserGrid1DBoundaryCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 Index gridXIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename Grid::Cell CellTopology;

   const IndexType& xSize = grid->getDimensions().x();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x );

   if( cellCoordinates.x() < grid->getDimensions().x() )
   {
      if( grid->isBoundaryCell( cellCoordinates ) )
      {
         EntitiesProcessor::template processEntity< CellTopology >
            ( *grid,
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
__global__ void tnlTraverserGrid1DInteriorCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename Grid::Cell CellTopology;

   CoordinatesType cellCoordinates;
   cellCoordinates.x() = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( cellCoordinates.x() > 0 && cellCoordinates.x() < grid->getDimensions().x() - 1 )
   {
      const IndexType index = grid->getCellIndex( cellCoordinates );
      EntitiesProcessor::template processEntity< CellTopology >( *grid, *userData, index, cellCoordinates );
   }
}

#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, 1 >::
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
      tnlTraverserGrid1DBoundaryCells< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
   cudaThreadSynchronize();
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}
template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Interior cells
    */
   checkCudaDevice;
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   dim3 cudaGridSize;
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx++ )
   {
      if( gridXIdx == cudaXGrids - 1 )
         cudaGridSize.x = cudaBlocks.x % tnlCuda::getMaxGridSize();
      tnlTraverserGrid1DInteriorCells< Real, Index, UserData, EntitiesProcessor >
                                     <<< cudaGridSize, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
   }
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor >
__global__ void tnlTraverserGrid1DBoundaryVertices( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                    UserData* userData,
                                                    Index gridXIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename Grid::Vertex VertexTopology;

   const IndexType& xSize = grid->getDimensions().x();

   CoordinatesType vertexCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x );

   if( vertexCoordinates.x() <= grid->getDimensions().x() )
   {
      if( grid->isBoundaryVertex( vertexCoordinates ) )
      {
         EntitiesProcessor::template processEntity< VertexTopology >
            ( *grid,
              *userData,
              grid->getVertexIndex( vertexCoordinates ),
              vertexCoordinates );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor >
__global__ void tnlTraverserGrid1DInteriorVertices( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                    UserData* userData,
                                                    const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename Grid::Vertex VertexTopology;

   CoordinatesType vertexCoordinates;
   vertexCoordinates.x() = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( vertexCoordinates.x() > 0 && vertexCoordinates.x() < grid->getDimensions().x() )
   {
      const IndexType index = grid->getVertexIndex( vertexCoordinates );
      EntitiesProcessor::template processEntity< CellTopology >( *grid, *userData, index, cellCoordinates );
   }
}
#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   #ifdef HAVE_CUDA

   /****
    * Boundary vertices
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      tnlTraverserGrid1DBoundaryVertices< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx );
   cudaThreadSynchronize();
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif

}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 1, Real, tnlCuda, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Interior vertices
    */
   checkCudaDevice;
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   dim3 cudaGridSize;
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx++ )
   {
      if( gridXIdx == cudaXGrids - 1 )
         cudaGridSize.x = cudaBlocks.x % tnlCuda::getMaxGridSize();
      tnlTraverserGrid1DInteriorVertices< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaGridSize, cudaBlockSize >>>
                                        ( kernelGrid,
                                          kernelUserData,
                                          gridXIdx );
   }
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}

#endif /* TNLTRAVERSER_GRID1D_IMPL_H_ */
