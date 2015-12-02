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
   typename GridType::template GridEntity< CellDimensions > entity( grid );

   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();   
   coordinates.x() = 0;
   entity.setIndex( grid.getEntityIndex( entity ) );
   EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   coordinates.x() = xSize - 1;
   entity.setIndex( grid.getEntityIndex( entity ) );
   EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
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
   typename GridType::template GridEntity< CellDimensions > cell( grid );

   const IndexType& xSize = grid.getDimensions().x();
   for( cell.getCoordinates().x() = 1;
        cell.getCoordinates().x() < xSize - 1;
        cell.getCoordinates().x()++ )
   {
      cell.setIndex( grid.getEntityIndex( cell ) );
      EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( cell ), cell );
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
   typename GridType::template GridEntity< VerticesDimensions > entity( grid );
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   coordinates.x() = 0;
   entity.setIndex( grid.getEntityIndex( entity ) );
   EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   coordinates.x() = xSize;
   entity.setIndex( grid.getEntityIndex( entity ) );
   EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
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
   typename GridType::template GridEntity< VerticesDimensions > entity( grid );
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
   {
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
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
   typename GridType::template GridEntity< CellDimensions > entity( grid );
   CoordinatesType& coordinates = entity.getCoordinates();

   const Index index = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.x() = Index;
   entity.setIndex( index )

   if( coordinates.x() < grid->getDimensions().x() )
   {
      if( processAllEntties || entity.isBoundaryEntity() == processBoundaryEntities )
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
   typename template GridEntity< GridfType::Vertices > vertex;

   const IndexType& xSize = grid->getDimensions().x();

   const Index index = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   vertex.getCoordinates().x() = index;
   vertex.setIndex( index );

   if( vertex.getCoordinates().x() <= grid->getDimensions().x() )
   {
      if( processAllEntities || vertex.isBoundaryEntity() == processBoundaryEntities )
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
