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
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

   for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
   {
      coordinates.y() = 0;
      EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
      coordinates.y() = ySize - 1;
      EntitiesProcessor::processCell( grid, userData, grid.getCellIndex( coordinates ), coordinates );
   }
   for( coordinates.y() = 1; coordinates.y() < ySize - 1; coordinates.y() ++ )
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
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior cells
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif
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
tnlTraverser< tnlGrid< 2, Real, tnlHost, Index >, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing boundary faces
    */
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

   for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
   {
      coordinates.y() = 0;
      EntitiesProcessor::processFace( grid, userData, grid.template getFaceIndex< 0, 1 >( coordinates ), coordinates );
      //cout << "Boundary face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 0, 1 >( coordinates ) << endl;
      coordinates.y() = ySize;
      EntitiesProcessor::processFace( grid, userData, grid.template getFaceIndex< 0, 1 >( coordinates ), coordinates );
      //cout << "Boundary face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 0, 1 >( coordinates ) << endl;
   }
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
   {
      coordinates.x() = 0;
      EntitiesProcessor::processFace( grid, userData, grid.template getFaceIndex< 1, 0 >( coordinates ), coordinates );
      //cout << "Boundary face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 1, 0 >( coordinates ) << endl;
      coordinates.x() = xSize;
      EntitiesProcessor::processFace( grid, userData, grid.template getFaceIndex< 1, 0 >( coordinates ), coordinates );
      //cout << "Boundary face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 1, 0 >( coordinates ) << endl;
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
   CoordinatesType coordinates;
   const IndexType& xSize = grid.getDimensions().x();
   const IndexType& ySize = grid.getDimensions().y();

#ifdef HAVE_OPENMP
//#pragma omp parallel for
#endif

   //cout << "< 1, 0 >" << endl;
   for( coordinates.y() = 0; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 1; coordinates.x() < xSize; coordinates.x() ++ )
      {
         const IndexType index = grid.template getFaceIndex< 1, 0 >( coordinates );
         EntitiesProcessor::processFace( grid, userData, index, coordinates );
         //cout << "Interior face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 1, 0 >( coordinates ) << endl;
      }

   //cout << "<  0, 1 >" << endl;
   for( coordinates.y() = 1; coordinates.y() < ySize; coordinates.y() ++ )
      for( coordinates.x() = 0; coordinates.x() < xSize; coordinates.x() ++ )
      {
         const IndexType index = grid.template getFaceIndex< 0, 1 >( coordinates );
         EntitiesProcessor::processFace( grid, userData, index, coordinates );
         //cout << "Interior face coordinates = " << coordinates << " index = " << grid.template getFaceIndex< 0, 1 >( coordinates ) << endl;
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
__global__ void tnlTraverserGrid2DBoundaryCells( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y );

   if( cellCoordinates.x() < grid->getDimensions().x() &&
       cellCoordinates.y() < grid->getDimensions().y() )
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
__global__ void tnlTraverserGrid2DInteriorCells( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();

   CoordinatesType cellCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y );

   if( cellCoordinates.x() < grid->getDimensions().x() &&
       cellCoordinates.y() < grid->getDimensions().y() )
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
          typename EntitiesProcessor,
          int nx,
          int ny >
__global__ void tnlTraverserGrid2DBoundaryFaces( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();

   CoordinatesType faceCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y );

   if( faceCoordinates.x() < grid->getDimensions().x() + nx &&
       faceCoordinates.y() < grid->getDimensions().y() + ny )
   {
      if( grid->isBoundaryFace( faceCoordinates ) )
      {
         //printf( "Processing boundary conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         EntitiesProcessor::processFace( *grid,
                                         *userData,
                                         grid->template getFaceIndex< nx, ny >( faceCoordinates ),
                                         faceCoordinates );
      }
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename EntitiesProcessor,
          int nx,
          int ny >
__global__ void tnlTraverserGrid2DInteriorFaces( const tnlGrid< 2, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 const Index gridXIdx,
                                                 const Index gridYIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   const IndexType& xSize = grid->getDimensions().x();
   const IndexType& ySize = grid->getDimensions().y();

   CoordinatesType faceCoordinates( ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x,
                                    ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y );

   if( faceCoordinates.x() < grid->getDimensions().x() + nx &&
       faceCoordinates.y() < grid->getDimensions().y() + ny )
   {
      if( ! grid->isBoundaryFace( faceCoordinates ) )
      {
         //printf( "Processing interior conditions at %d %d \n", cellCoordinates.x(), cellCoordinates.y() );
         EntitiesProcessor::processFace( *grid,
                                         *userData,
                                         grid->template getFaceIndex< nx, ny >( faceCoordinates ),
                                         faceCoordinates );
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
         tnlTraverserGrid2DBoundaryCells< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
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
         tnlTraverserGrid2DInteriorCells< Real, Index, UserData, EntitiesProcessor >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
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
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 1 >::
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
         tnlTraverserGrid2DBoundaryFaces< Real, Index, UserData, EntitiesProcessor, 1, 0 >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
      }
   cudaThreadSynchronize();
   checkCudaDevice;

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
         tnlTraverserGrid2DBoundaryFaces< Real, Index, UserData, EntitiesProcessor, 0, 1 >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
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
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Traversing interior faces
    */
#ifdef HAVE_CUDA

   /****
    * Boundary conditions
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
         tnlTraverserGrid2DInteriorFaces< Real, Index, UserData, EntitiesProcessor, 1, 0 >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
      }
   cudaThreadSynchronize();
   checkCudaDevice;

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
         tnlTraverserGrid2DInteriorFaces< Real, Index, UserData, EntitiesProcessor, 0, 1 >
                                        <<< cudaBlocks, cudaBlockSize >>>
                                       ( kernelGrid,
                                         kernelUserData,
                                         gridXIdx,
                                         gridYIdx );
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
tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary interior vertices
    */
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
   /****
    * Traversing interior vertices
    */
}


#endif /* TNLTRAVERSER_GRID2D_IMPL_H_ */
