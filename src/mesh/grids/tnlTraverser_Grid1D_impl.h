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
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity;

   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
   coordinates.x() = xSize - 1;
   EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
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
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity;

   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize-1; coordinates.x() ++ )
   {
      const IndexType index = grid.getCellIndex( coordinates );
      EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
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
   const int VerticesDimensions = 0;
   typename GridType::template GridEntity< VerticesDimensions > entity;
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
   coordinates.x() = xSize;
   EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
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
   const int VerticesDimensions = 0;
   typename GridType::template GridEntity< VerticesDimensions > entity;
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
   {
      EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
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
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid1DCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         Index gridXIdx )
{
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity;
   CoordinatesType& coordinates = entity.getCoordinates();


   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;

   if( coordinates.x() < grid->getDimensions().x() )
   {
      if( processAllEntties || grid->isBoundaryEntity( entity ) == processBoundaryEntities )
      {
         EntitiesProcessor::processEntity( *grid, *userData, grid->getEntityIndex( entity ), entity );
      }
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
      tnlTraverserGrid1DCells< Real, Index, UserData, EntitiesProcessor, false, true >
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
      tnlTraverserGrid1DCells< Real, Index, UserData, EntitiesProcessor, false, false >
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
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid1DVertices( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                            UserData* userData,
                                            Index gridXIdx )
{
   typedef typename GridType::Vertex VertexTopology;

   const IndexType& xSize = grid->getDimensions().x();

   CoordinatesType vertexCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x );

   if( vertexCoordinates.x() <= grid->getDimensions().x() )
   {
      if( processAllEntities || grid->isBoundaryVertex( vertexCoordinates ) == processBoundaryEntities )
      {
         EntitiesProcessor::template processEntity< VertexTopology >
            ( *grid,
              *userData,
              grid->getVertexIndex( vertexCoordinates ),
              vertexCoordinates );
      }
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
      tnlTraverserGrid1DVertices< Real, Index, UserData, EntitiesProcessor, false, true >
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
      tnlTraverserGrid1DInteriorVertices< Real, Index, UserData, EntitiesProcessor, false, false >
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
