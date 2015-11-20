/***************************************************************************
                          tnlTraverser_Grid3D_impl.h  -  description
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

#ifndef TNLTRAVERSER_GRID3D_IMPL_H_
#define TNLTRAVERSER_GRID3D_IMPL_H_

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 3 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary conditions
    */
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity;
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.z() = zSize - 1;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.y() = ySize - 1;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.x() = xSize - 1;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
}
template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 3 >::
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
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.z() = 1; coordinates.z() < zSize - 1; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize - 1; coordinates.x() ++ )
         {
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary faces
    */
   const int FaceDimensions = GridType::Dimensions - 1;
   typedef typename GridType::template GridEntity< FaceDimensions > EntityType;
   EntityType entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   entity.setOrientation( typename EntityType::EntityOrientationType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.x() = xSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   entity.setOrientation( typename EntityType::EntityOrientationType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.y() = ySize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
  
   entity.setOrientation( typename EntityType::EntityOrientationType( 0, 0, 1 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )      
      {         
         coordinates.z() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.z() = zSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }     
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior faces
    */
   const int FaceDimensions = GridType::Dimensions - 1;
   typedef typename GridType::template GridEntity< FaceDimensions > EntityType;
   EntityType entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   entity.setOrientation( typename EntityType::EntityOrientationType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );

   entity.setOrientation( typename EntityType::EntityOrientationType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
   
   entity.setOrientation( typename EntityType::EntityOrientationType( 0, 0, 1 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
}


template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary edges
    */
   const int EdgeDimensions = 1;
   typedef typename GridType::template GridEntity< EdgeDimensions > EntityType;
   EntityType entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();
   
   entity.setBasis( typename EntityType::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )      
      {
         coordinates.z() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );        
         coordinates.z() = zSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
   
   entity.setBasis( typename EntityType::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )      
      {
         coordinates.z() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.z() = zSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
   
   entity.setBasis( typename EntityType::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.y() = ySize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
   
   entity.setBasis( typename EntityType::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.y() = ySize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   entity.setBasis( typename EntityType::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.x() = xSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
   
   entity.setBasis( typename EntityType::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEdgeIndex( entity ), entity );
         coordinates.x() = xSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEdgeIndex( entity ), entity );
      }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior edges
    */
   const int EdgeDimensions = 1;
   typedef typename GridType::template GridEntity< EdgeDimensions > EntityType;
   EntityType entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   entity.setBasis( typename EntityType::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );

   entity.setBasis( typename EntityType::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
   
   entity.setBasis( typename EntityType::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
}


template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary vertices
    */
   const int VertexDimensions = 0;
   typedef typename GridType::template GridEntity< VertexDimensions > EntityType;
   EntityType entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();
   typedef typename GridType::Vertex VertexTopology;
   
   for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.z() = zSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.y() = ySize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
         coordinates.x() = xSize;
         EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
      }
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior vertices
    */
   const int VertexDimensions = 0;
   typedef typename GridType::template GridEntity< VertexDimensions > EntityType;
   EntityType entity;
   
   CoordinatesType& coordinates = entity.getCoordinates();
   
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
            EntitiesProcessor::processEntity( grid, userData, grid.getEntityIndex( entity ), entity );
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
__global__ void tnlTraverserGrid3DCells( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         const Index gridXIdx,
                                         const Index gridYIdx,
                                         const Index gridZIdx )
{
   const int CellDimensions = GridType::Dimensions;
   typename GridType::template GridEntity< CellDimensions > entity;
   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();
   const IndexType& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() &&
       coordinates.y() < grid->getDimensions().y() &&
       coordinates.z() < grid->getDimensions().z() )
   {
      if( processAllEntities || grid->isBoundaryEntity( entity ) == processBoundaryEntities )
      {
         //printf( "Processing boundary conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         EntitiesProcessor::template processEntity
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
__global__ void tnlTraverserGrid3DFaces( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         const Index gridXIdx,
                                         const Index gridYIdx,
                                         const Index gridZIdx,
                                         int nx,
                                         int ny,
                                         int nz )
{
   const int FaceDimensions = GridType::Dimensions - 1;
   typename GridType::template GridEntity< FaceDimensions > entity;
   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();
   const IndexType& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() + nx &&
       coordinates.y() < grid->getDimensions().y() + ny &&
       coordinates.z() < grid->getDimensions().z() + nz )
   {
      if( processAllEntities || grid->isBoundaryEntity( entity ) == processBoundaryEntities )
      {
         //printf( "Processing boundary conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         EntitiesProcessor::processEntity
            ( *grid,
              *userData,
              grid->geEntityIndex( entity ),
              entity,
              nx, ny, nz );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor,
          bool processAllEntities,
          bool processBoundaryEntities >
__global__ void tnlTraverserGrid3DEdges( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                         UserData* userData,
                                         const Index gridXIdx,
                                         const Index gridYIdx,
                                         const Index gridZIdx,
                                         int dx,
                                         int dy,
                                         int dz )
{
   const int EdgeDimensions = 1;
   typename GridType::template GridEntity< EdgeDimensions > entity;
   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();
   const IndexType& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() + !nx &&
       coordinates.y() < grid->getDimensions().y() + !ny &&
       coordinates.z() < grid->getDimensions().z() + !nz )
   {
      if( processAllEntities || grid->isBoundaryEntity( entity ) == processBoundaryEntity )
      {
         //printf( "Processing boundary conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
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
__global__ void tnlTraverserGrid3DVertices( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
                                            UserData* userData,
                                            const Index gridXIdx,
                                            const Index gridYIdx,
                                            const Index gridZIdx )
{
   const int VertexDimensions = 0;
   typename GridType::template GridEntity< VertexDimensions > entity;
   CoordinatesType& coordinates = entity.getCoordinates();
   
   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();
   const IndexType& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() &&
       coordinates.y() < grid->getDimensions().y() &&
       coordinates.z() < grid->getDimensions().z() )
   {
      if( processAllEntities || grid->isBoundaryVertex( vertexCoordinates ) == processBoundaryEntities )
      {
         EntitiesProcessor::template processEntity
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
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 3 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary cells
    */
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

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DCells< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx );
         }
   cudaThreadSynchronize();
   checkCudaDevice;
#endif
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 3 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   #ifdef HAVE_CUDA

   /****
    * Interior cells
    */
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

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DCells< Real, Index, UserData, EntitiesProcessor, false, false >
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
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary faces
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids, cudaZGrids;

   /****
    * < 1, 0, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DFaces< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 1, 0, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 1, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DFaces< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 1, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 0, 1 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DFaces< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 0, 1 );
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
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Traversing interior faces
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids, cudaZGrids;

   /****
    * < 1, 0, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DFaces< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 1, 0, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 1, 0 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DFaces< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 1, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 0, 1 > faces
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DInteriorFaces< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 0, 1 );
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
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary edges
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids, cudaZGrids;

   /****
    * < 1, 0, 0 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 1, 0, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 1, 0 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 1, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 0, 1 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 0, 1 );
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
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary edges
    */
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   IndexType cudaXGrids, cudaYGrids, cudaZGrids;

   /****
    * < 1, 0, 0 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x(), cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 1, 0, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 1, 0 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y(), cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 1, 0 );
            checkCudaDevice;
         }
   cudaThreadSynchronize();
   
   /****
    * < 0, 0, 1 > edges
    */
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z(), cudaBlockSize.z );
   cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z);
   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DEdges< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 0, 0, 1 );
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
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary vertices
    */
#ifdef HAVE_CUDA
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DVertices< Real, Index, UserData, EntitiesProcessor, false, true >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx );
         }
   cudaThreadSynchronize();
   checkCudaDevice;
#endif
   
}

template< typename Real,
          typename Index >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior vertices
    */
#ifdef HAVE_CUDA
   GridType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );

   dim3 cudaBlockSize( 8, 8, 4 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( grid.getDimensions().x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( grid.getDimensions().y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( grid.getDimensions().z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
         {
            tnlTraverserGrid3DInteriorVertices< Real, Index, UserData, EntitiesProcessor, false, false >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
                 kernelUserData,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx );
         }
   cudaThreadSynchronize();
   checkCudaDevice;
#endif
}


#endif /* TNLTRAVERSER_GRID3D_IMPL_H_ */
