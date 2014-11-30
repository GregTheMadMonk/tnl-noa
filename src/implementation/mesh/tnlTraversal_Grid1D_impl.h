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
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitiesProcessor ) const
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
   boundaryEntitiesProcessor.processCell( grid, userData, 0, coordinates );
   coordinates.x() = xSize - 1;
   boundaryEntitiesProcessor.processCell( grid, userData, xSize - 1, coordinates );

   /****
    * Interior cells
    */
   for( coordinates.x() = 1; coordinates.x() < xSize-1; coordinates.x() ++ )
   {
      const IndexType index = grid.getCellIndex( coordinates );
      interiorEntitiesProcessor.processCell( grid, userData, index, coordinates );
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
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitiesProcessor ) const
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
         boundaryEntitiesProcessor.processVertices( grid, userData, index, coordinates );
      else
         interiorEntitiesProcessor.processVertices( grid, userData, index, coordinates );
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename UserData,
          typename BoundaryEntitiesProcessor >
__global__ void tnlTraversalGrid1DBoundaryCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 BoundaryEntitiesProcessor* boundaryEntitiesProcessor )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   CoordinatesType coordinates;
   const IndexType& xSize = grid->getDimensions().x();

   if( threadIdx.x == 0 )
   {
      coordinates.x() = 0;
      boundaryEntitiesProcessor->processCell( *grid, *userData, 0, coordinates );
   }
   else
   {
      coordinates.x() = xSize - 1;
      boundaryEntitiesProcessor->processCell( *grid, *userData, xSize - 1, coordinates );
   }
}

template< typename Real,
          typename Index,
          typename UserData,
          typename InteriorEntitiesProcessor >
__global__ void tnlTraversalGrid1DInteriorCells( const tnlGrid< 1, Real, tnlCuda, Index >* grid,
                                                 UserData* userData,
                                                 InteriorEntitiesProcessor* interiorEntitiesProcessor,
                                                 int gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;

   CoordinatesType coordinates;
   coordinates.x() = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( coordinates.x() != 0 && coordinates.x() != grid->getDimensions().x() - 1 )
   {
      const IndexType index = grid->getCellIndex( coordinates );
      interiorEntitiesProcessor->processCell( *grid, *userData, index, coordinates );
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
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitiesProcessor ) const
{
   /****
    * Traversing cells
    */
#ifdef HAVE_CUDA

   dim3 cudaBlockSize( 256 );

   /****
    * Boundary conditions
    */
   cerr << "Setting boundary conditions." << endl;
   tnlTraversalGrid1DBoundaryCells<<< 1, 2 >>>
                                    ( &grid,
                                      &userData,
                                      &boundaryEntitiesProcessor );
   cudaThreadSynchronize();
   checkCudaDevice;
   cerr << "Setting boundary conditions... done" << endl;
   return;

   /****
    * Interior cells
    */
   const IndexType cudaBlocks = tnlCuda::getNumberOfBlocks( grid.getNumberOfCells(), cudaBlockSize.x );
   const IndexType cudaGrids = tnlCuda::getNumberOfGrids( cudaBlocks );
   dim3 cudaGridSize;
   cerr << "Evaluating operator..." << endl;
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
      tnlTraversalGrid1DInteriorCells<<< cudaGridSize, cudaBlockSize >>>
                                       ( &grid,
                                         &userData,
                                         &interiorEntitiesProcessor,
                                         gridIdx );
   }
   checkCudaDevice;
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
                 UserData& userData,
                 BoundaryEntitiesProcessor& boundaryEntitiesProcessor,
                 InteriorEntitiesProcessor& interiorEntitesProcessor ) const
{
   /****
    * Traversing vertices
    */

}



#endif /* TNLTRAVERSAL_GRID1D_IMPL_H_ */
