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
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.z() = zSize - 1;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.y() = ySize - 1;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() < zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
         coordinates.x() = xSize - 1;
         EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
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
   CoordinatesType coordinates;
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
            const IndexType index = grid.getCellIndex( coordinates );
            EntitiesProcessor::processCell( grid, userData, index, coordinates );
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
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();
   const IndexType& zSize = grid.getDimensions().z();

   for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.z() = 0;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
         coordinates.z() = zSize;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.x() = 0; coordinates.x() <= xSize; coordinates.x() ++ )
      {
         coordinates.y() = 0;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
         coordinates.y() = ySize;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
      }

   for( coordinates.z() = 0; coordinates.z() <= zSize; coordinates.z() ++ )
      for( coordinates.y() = 0; coordinates.y() <= ySize; coordinates.y() ++ )
      {
         coordinates.x() = 0;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
         coordinates.x() = xSize;
         EntitiesProcessor::processVertex( grid, userData, grid.getVertexIndex( coordinates ), coordinates );
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
   CoordinatesType coordinates;
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
            const IndexType index = grid.getVertexIndex( coordinates );
            EntitiesProcessor::processVertex( grid, userData, index, coordinates );
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
          typename EntitiesProcessor >
__global__ void tnlTraverserGrid3DBoundaryCells( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
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
__global__ void tnlTraverserGrid3DInteriorCells( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
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
__global__ void tnlTraverserGrid3DBoundaryVertices( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
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

   CoordinatesType vertexCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                      ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y,
                                      ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z );

   if( vertexCoordinates.x() < grid->getDimensions().x() &&
       vertexCoordinates.y() < grid->getDimensions().y() &&
       vertexCoordinates.z() < grid->getDimensions().z() )
   {
      if( grid->isBoundaryVertex( vertexCoordinates ) )
      {
         EntitiesProcessor::processVertex( *grid,
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
__global__ void tnlTraverserGrid3DInteriorVertices( const tnlGrid< 3, Real, tnlCuda, Index >* grid,
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

   CoordinatesType vertexCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                      ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y,
                                      ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z );

   if( vertexCoordinates.x() < grid->getDimensions().x() &&
       vertexCoordinates.y() < grid->getDimensions().y() &&
       vertexCoordinates.z() < grid->getDimensions().z() )
   {
      if( ! grid->isBoundaryVertex( vertexCoordinates ) )
      {
         EntitiesProcessor::processVertex( *grid,
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
            tnlTraverserGrid3DBoundaryCells< Real, Index, UserData, EntitiesProcessor >
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
            tnlTraverserGrid3DInteriorCells< Real, Index, UserData, EntitiesProcessor >
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
   /****
    * Traversing boundary faces
    */
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
   /****
    * Traversing interior faces
    */
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
   /****
    * Traversing boundary edges
    */
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
   /****
    * Traversing interior edges
    */
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
            tnlTraverserGrid3DBoundaryVertices< Real, Index, UserData, EntitiesProcessor >
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
            tnlTraverserGrid3DInteriorVertices< Real, Index, UserData, EntitiesProcessor >
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
