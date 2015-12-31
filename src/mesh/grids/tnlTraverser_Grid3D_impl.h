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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 3 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary conditions
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
   GridEntity entity( grid );
   
   CoordinatesType& coordinates = entity.getCoordinates();
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.z() = zSize - 1;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.y() = ySize - 1;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.x() = xSize - 1;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
}
template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 3 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
   GridEntity entity( grid );

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
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }
}

template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
   GridEntity entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   entity.setOrientation( typename GridEntity::EntityOrientationType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.x() = xSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   entity.setOrientation( typename GridEntity::EntityOrientationType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.y() = ySize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
  
   entity.setOrientation( typename GridEntity::EntityOrientationType( 0, 0, 1 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )      
      {         
         coordinates.z() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.z() = zSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }     
}

template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
   GridEntity entity( grid );

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   entity.setOrientation( typename GridEntity::EntityOrientationType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();         
            EntitiesProcessor::processEntity( grid, userData, entity );
         }

   entity.setOrientation( typename GridEntity::EntityOrientationType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }
            
   
   entity.setOrientation( typename GridEntity::EntityOrientationType( 0, 0, 1 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }
}


template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
   GridEntity entity( grid );

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();
   
   entity.setBasis( typename GridEntity::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )      
      {
         coordinates.z() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );        
         coordinates.z() = zSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
   
   entity.setBasis( typename GridEntity::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )      
      {
         coordinates.z() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.z() = zSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
   
   entity.setBasis( typename GridEntity::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.y() = ySize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
   
   entity.setBasis( typename GridEntity::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.y() = ySize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   entity.setBasis( typename GridEntity::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.x() = xSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
   
   entity.setBasis( typename GridEntity::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.x() = xSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
}

template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );   
   GridEntity entity( grid );

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   entity.setBasis( typename GridEntity::EntityBasisType( 0, 0, 1 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }

   entity.setBasis( typename GridEntity::EntityBasisType( 0, 1, 0 ) );
   for( coordinates.z() = 1; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }
   
   entity.setBasis( typename GridEntity::EntityBasisType( 1, 0, 0 ) );
   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
         for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
         }
}


template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
   GridEntity entity;

   CoordinatesType& coordinates = entity.getCoordinates();

   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();
   typedef typename GridType::Vertex VertexTopology;
   
   for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.z() = zSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.y() = ySize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
         coordinates.x() = xSize;
         entity.refresh();
         EntitiesProcessor::processEntity( grid, userData, entity );
      }
}

template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlHost, Index >, GridEntity, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
   GridEntity entity( grid );
   
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
         {
            entity.refresh();
            EntitiesProcessor::processEntity( grid, userData, entity );
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
          typename GridEntity, 
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
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   GridEntity entity( *grid );
   typedef typename GridType::CoordinatesType CoordinatesType;
   CoordinatesType& coordinates = entity.getCoordinates();

   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();
   const Index& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() &&
       coordinates.y() < grid->getDimensions().y() &&
       coordinates.z() < grid->getDimensions().z() )
   {
      entity.refresh();
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::template processEntity
            ( *grid,
              *userData,
              entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename GridEntity, 
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
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   
   GridEntity entity( *grid );
   entity.setOrientation( nx, ny, nz );   
   CoordinatesType& coordinates = entity.getCoordinates();

   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();
   const Index& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() + nx &&
       coordinates.y() < grid->getDimensions().y() + ny &&
       coordinates.z() < grid->getDimensions().z() + nz )
   {
      entity.refresh();
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::processEntity
            ( *grid,
              *userData,
              entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename GridEntity, 
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
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   
   GridEntity entity( *grid );
   entity.setBasis( dx, dy, dz );
   
   CoordinatesType& coordinates = entity.getCoordinates();

   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();
   const Index& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() + !dx &&
       coordinates.y() < grid->getDimensions().y() + !dy &&
       coordinates.z() < grid->getDimensions().z() + !dz )
   {
      entity.refresh();
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::processEntity
            ( *grid,
              *userData,
              entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename GridEntity, 
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
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   GridEntity entity( *grid );
   CoordinatesType& coordinates = entity.getCoordinates();
   
   const Index& xSize = grid->getDimensions().x();
   const Index& ySize = grid->getDimensions().y();
   const Index& zSize = grid->getDimensions().z();

   coordinates.x() = ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates.x() < grid->getDimensions().x() &&
       coordinates.y() < grid->getDimensions().y() &&
       coordinates.z() < grid->getDimensions().z() )
   {
      entity.refresh();
      if( processAllEntities || entity.isBoundaryEntity() == processBoundaryEntities )
      {         
         EntitiesProcessor::template processEntity
         ( *grid,
           *userData,
           entity );
      }
   }
}

#endif

template< typename Real,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 3 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DCells< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 3 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   #ifdef HAVE_CUDA

   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DCells< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Traversing interior faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
            tnlTraverserGrid3DFaces< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
#ifdef HAVE_CUDA
   /****
    * Boundary edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
            tnlTraverserGrid3DEdges< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary vertices
    */
#ifdef HAVE_CUDA
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DVertices< Real, Index, GridEntity, UserData, EntitiesProcessor, false, true >
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
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, tnlCuda, Index >, GridEntity, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior vertices
    */
#ifdef HAVE_CUDA
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
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
            tnlTraverserGrid3DVertices< Real, Index, GridEntity, UserData, EntitiesProcessor, false, false >
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
