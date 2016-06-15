/***************************************************************************
                          tnlGridTraverser_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLGRIDTRAVERSER_IMPL_H
#define	TNLGRIDTRAVERSER_IMPL_H

/****
 * 1D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
tnlGridTraverser< tnlGrid< 1, Real, tnlHost, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,   
   UserData& userData )
{

   
   GridEntity entity( *gridPointer );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );
   if( processOnlyBoundaryEntities )
   {
      entity.getCoordinates() = begin;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
      entity.getCoordinates() = end;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
   else
   {
      for( entity.getCoordinates().x() = begin.x();
           entity.getCoordinates().x() <= end.x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
      }
   }
}

/****
 * 1D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void 
tnlGridTraverser1D(
   const tnlGrid< 1, Real, tnlCuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,   
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
   
   coordinates.x() = begin->x() + ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   
   GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );
   
   entity.refresh();
   if( coordinates.x() <= end->x() )
   {
      //if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {         
         EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void 
tnlGridBoundaryTraverser1D(
   const tnlGrid< 1, Real, tnlCuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
   
   if( threadIdx.x == 0 )
   {   
      coordinates.x() = begin->x();
      GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
   }
   if( threadIdx.x == 1 )
   {   
      coordinates.x() = end->x();
      GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
   }
}

#endif

template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
tnlGridTraverser< tnlGrid< 1, Real, tnlCuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,   
   UserData& userData )
{
#ifdef HAVE_CUDA   
   CoordinatesType* kernelBegin = tnlCuda::passToDevice( begin );
   CoordinatesType* kernelEnd = tnlCuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = tnlCuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = tnlCuda::passToDevice( entityBasis );
   //typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( *gridPointer );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
      
   tnlCuda::synchronizeDevice();
   if( processOnlyBoundaryEntities )
   {
      dim3 cudaBlockSize( 2 );
      dim3 cudaBlocks( 1 );
      tnlGridBoundaryTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize >>>
            ( &gridPointer.template getData< tnlCuda >(),
              kernelUserData,
              kernelBegin,
              kernelEnd,
              kernelEntityOrientation,
              kernelEntityBasis );
   }
   else
   {
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
      const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         tnlGridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize >>>
            ( &gridPointer.template getData< tnlCuda >(),
              kernelUserData,
              kernelBegin,
              kernelEnd,
              kernelEntityOrientation,
              kernelEntityBasis,
              gridXIdx );
   }
   cudaThreadSynchronize();
   checkCudaDevice;
   //tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}


/****
 * 2D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 2, Real, tnlHost, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
   GridEntity entity( *gridPointer );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );

   if( processOnlyBoundaryEntities )
   {
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y() = begin.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().y() = end.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
         {
            entity.getCoordinates().x() = begin.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().x() = end.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
   }
   else
   {
//#pragma omp parallel for firstprivate( entity, begin, end ) if( tnlHost::isOMPEnabled() )      
      for( entity.getCoordinates().y() = begin.y();
           entity.getCoordinates().y() <= end.y();
           entity.getCoordinates().y() ++ )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
   }
}

/****
 * 2D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities >
__global__ void 
tnlGridTraverser2D(
   const tnlGrid< 2, Real, tnlCuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,   
   const Index gridXIdx,
   const Index gridYIdx )
{
   typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   //if( threadIdx.x == 0 )
   //   printf( "%d x %d \n ", grid->getDimensions().x(), grid->getDimensions().y() );
   
   coordinates.x() = begin->x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin->y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   
   GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );
   
   if( entity.getCoordinates().x() <= end->x() &&
       entity.getCoordinates().y() <= end->y() )
   {
      entity.refresh();
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {         
         EntitiesProcessor::processEntity
         ( *grid,
           *userData,
           entity );
      }
   }
}
#endif

template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary,
         int YOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 2, Real, tnlCuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
#ifdef HAVE_CUDA   
   CoordinatesType* kernelBegin = tnlCuda::passToDevice( begin );
   CoordinatesType* kernelEnd = tnlCuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = tnlCuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = tnlCuda::passToDevice( entityBasis );
   //typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( *gridPointer );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
   checkCudaDevice;   
      
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   
   tnlCuda::synchronizeDevice();
   for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         tnlGridTraverser2D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities >
            <<< cudaBlocks, cudaBlockSize >>>
            ( &gridPointer.template getData< tnlCuda >(),
              kernelUserData,
              kernelBegin,
              kernelEnd,
              kernelEntityOrientation,
              kernelEntityBasis,
              gridXIdx,
              gridYIdx );
      
   cudaThreadSynchronize();
   checkCudaDevice;
   //tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}

/****
 * 3D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary,
      int ZOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 3, Real, tnlHost, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
   GridEntity entity( *gridPointer );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );

   if( processOnlyBoundaryEntities )
   {
      if( ZOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().z() = begin.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().z() = end.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
                 entity.getCoordinates().z() <= end.z();
                 entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().y() = begin.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().y() = end.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
              entity.getCoordinates().z() <= end.z();
              entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().y() = begin.y();
                 entity.getCoordinates().y() <= end.y();
                 entity.getCoordinates().y() ++ )
            {
               entity.getCoordinates().x() = begin.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().x() = end.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
   }
   else
   {
      for( entity.getCoordinates().z() = begin.z();
           entity.getCoordinates().z() <= end.z();
           entity.getCoordinates().z() ++ )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
   }
}

/****
 * 3D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities >
__global__ void 
tnlGridTraverser3D(
   const tnlGrid< 3, Real, tnlCuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,   
   const Index gridXIdx,
   const Index gridYIdx,
   const Index gridZIdx )
{
   typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin->x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin->y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   coordinates.z() = begin->z() + ( gridZIdx * tnlCuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;  
   
   GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );

   if( entity.getCoordinates().x() <= end->x() &&
       entity.getCoordinates().y() <= end->y() && 
       entity.getCoordinates().z() <= end->z() )
   {
      entity.refresh();
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {         
         EntitiesProcessor::processEntity
         ( *grid,
           *userData,
           entity );
      }
   }
}
#endif

template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary,
         int YOrthogonalBoundary,
         int ZOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 3, Real, tnlCuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
#ifdef HAVE_CUDA   
   CoordinatesType* kernelBegin = tnlCuda::passToDevice( begin );
   CoordinatesType* kernelEnd = tnlCuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = tnlCuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = tnlCuda::passToDevice( entityBasis );
   //typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
      
   dim3 cudaBlockSize( 8, 8, 8 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( end.z() - begin.z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );

   tnlCuda::synchronizeDevice();
   for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
            tnlGridTraverser3D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities >
               <<< cudaBlocks, cudaBlockSize >>>
               ( &gridPointer.template getData< tnlCuda >(),
                 kernelUserData,
                 kernelBegin,
                 kernelEnd,
                 kernelEntityOrientation,
                 kernelEntityBasis,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx );
   cudaThreadSynchronize();
   checkCudaDevice;   
   //tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}


#endif	/* TNLGRIDTRAVERSER_IMPL_H */

