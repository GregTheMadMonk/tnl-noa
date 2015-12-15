/***************************************************************************
                          tnlTraverser_Grid2D_impl.h  -  description
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


#ifndef TNLTRAVERSER_GRID2D_IMPL_H_
#define TNLTRAVERSER_GRID2D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary cells
    */
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity( grid );

   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

   for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
   {
      coordinates.y() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.y() = ySize - 1;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }
   for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
   {
      coordinates.x() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.x() = xSize - 1;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior cells
    */
   const int CellDimensions = GridType::Dimensions;
   
   typename GridType::template GridEntity< CellDimensions > entity( grid );
   CoordinatesType& coordinates = entity.getCoordinates();
   
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
      for( coordinates.x() = 1; coordinates.x() < xSize - 1; coordinates.x() ++ )
      {
         entity.setIndex( grid.getEntityIndex( entity ) );
         EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary faces
    */   
   const int FaceDimensions = GridType::Dimensions - 1;
   typedef typename GridType::template GridEntity< FaceDimensions > EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientationType;
   EntityType entity( grid );

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

   CoordinatesType& coordinates = entity.getCoordinates();
   entity.setOrientation( EntityOrientationType( 1, 0 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
   {
      coordinates.x() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.x() = xSize;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }

   entity.setOrientation( EntityOrientationType( 0, 1 ) );
   for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
   {      
      coordinates.y() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.y() = ySize;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior faces
    */
   const int FaceDimensions = GridType::Dimensions - 1;
   typedef typename GridType::template GridEntity< FaceDimensions > EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientationType;
   EntityType entity( grid );

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   CoordinatesType& coordinates = entity.getCoordinates();
   entity.setOrientation( EntityOrientationType( 1, 0 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
      {
         entity.setIndex( grid.getEntityIndex( entity ) );
         EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      }

   entity.setOrientation( EntityOrientationType( 0, 1 ) );
   for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         entity.setIndex( grid.getEntityIndex( entity ) );
         EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      }
}


template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary vertices
    */
   const int VertexDimensions = 0;
   typedef typename GridType::template GridEntity< VertexDimensions > EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientation;
   EntityType entity( grid );

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   
   CoordinatesType& coordinates = entity.getCoordinates();
   for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
   {
      coordinates.y() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.y() = ySize;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }
   for( coordinates.y() = 1; coordinates.y() <= ySize; coordinates.y() ++ )
   {
      coordinates.x() = 0;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
      coordinates.x() = xSize;
      entity.setIndex( grid.getEntityIndex( entity ) );
      EntitiesProcessor::processEntity( grid, userData, entity.getIndex(), entity );
   }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior vertices
    */
   const int VertexDimensions = 0;
   typedef typename GridType::template GridEntity< VertexDimensions > EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientation;
   EntityType entity( grid );
   
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

#ifdef HAVE_OPENMP
  //#pragma omp parallel for
#endif
   CoordinatesType& coordinates = entity.getCoordinates();
   for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
      {
         entity.setIndex( grid.getEntityIndex( entity ) );
         EntitiesProcessor::processEntity( grid, userData, grid.getVertexIndex( entity ), entity );
      }  
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
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid2DCells( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         const Index gridXIdx,
                                         const Index gridYIdx )
{
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity( *grid );
   typedef typename GridType::CoordinatesType CoordinatesType;
   //CoordinatesType& coordinates = entity.getCoordinates();

   /*const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();*/

   entity.getCoordinates().x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   entity.getCoordinates().y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  

   if( entity.getCoordinates().x() < grid->getDimensions().x() &&
       entity.getCoordinates().y() < grid->getDimensions().y() )
   {
      entity.setIndex( grid->getEntityIndex( entity ) );
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::processEntity
         ( *grid,
           *userData,
           grid->getEntityIndex( entity ),
           entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid2DFaces( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         const Index gridXIdx,
                                         const Index gridYIdx,
                                         int nx,
                                         int ny )
{
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   const int FaceDimensions = GridType::Dimensions - 1;
   typedef typename GridType::template GridEntity< GridType::Cells > EntityType;
   EntityType entity( *grid );
   typedef typename GridType::CoordinatesType CoordinatesType;
   CoordinatesType& coordinates = entity.getCoordinates();
   entity.setOrientation( typename EntityType::EntityOrientationType( nx, ny ) );

   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;

   if( coordinates.x() < grid->getDimensions().x() + nx &&
       coordinates.y() < grid->getDimensions().y() + ny )
   {
      entity.setIndex( grid->getEntityIndex( entity ) );
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::processEntity
            ( *grid,
              *userData,
              grid->getEntityIndex( entity ),
              entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid2DVertices( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                            UserData* userData,
                                            const Index gridXIdx,
                                            const Index gridYIdx )
{
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   const int VertexDimensions = 0;
   typedef typename GridType::template GridEntity< VertexDimensions > EntityType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   EntityType entity( *grid );
   CoordinatesType& coordinates = entity.getCoordinates();

   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;

   if( coordinates.x() <= grid->getDimensions().x() &&
       coordinates.y() <= grid->getDimensions().y() )
   {
      entity.setIndex( grid->getEntityIndex( entity ) );
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {
         EntitiesProcessor::processEntity
         ( *grid,
           *userData,
           grid->getEntityIndex( entity ),
           entity );
      }
   }
}
#endif

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA

   /****
    * Boundary conditions
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DCells< Real, Index, UserData, EntitiesProcessor, false, true >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
         checkCudaDevice;
      }
   cudaThreadSynchronize();   
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Interior cells
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DCells< Real, Index, UserData, EntitiesProcessor, false, false >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
         checkCudaDevice;
      }
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelUserData );
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary faces
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids;

   /****
    * < 1, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DFaces< Real, Index, UserData, EntitiesProcessor, false, true >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx,
              1, 0  );
         checkCudaDevice;
      }
   cudaThreadSynchronize();
   

   /****
    * < 0, 1 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DFaces< Real, Index, UserData, EntitiesProcessor, false, true >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx,
              0, 1 );
         checkCudaDevice;
      }
   cudaThreadSynchronize();
   
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Traversing interior faces
    */
   
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids;

   /****
    * < 1, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DFaces< Real, Index, UserData, EntitiesProcessor, false, false >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx,
              1, 0 );
         checkCudaDevice;
      }
   cudaThreadSynchronize();
   

   /****
    * < 0, 1 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DFaces< Real, Index, UserData, EntitiesProcessor, false, false >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx,
              0, 1 );
         checkCudaDevice;
      }
   cudaThreadSynchronize();
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Traversing boundary vertices    
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DVertices< Real, Index, UserData, EntitiesProcessor, false, true >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx );
         checkCudaDevice;
      }
   cudaThreadSynchronize();   
#endif
}


template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Traversing interior vertices    
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      {
         tnlTraverserGrid2DVertices< Real, Index, UserData, EntitiesProcessor, false, false >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              gridXIdx,
              gridYIdx );
         checkCudaDevice;
      }
   cudaThreadSynchronize();   
#endif
}


#endif /* TNLTRAVERSER_GRID2D_IMPL_H_ */
