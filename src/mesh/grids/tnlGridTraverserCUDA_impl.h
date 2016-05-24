/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlGridTraverserCUDA_impl.h
 * Author: hanouvit
 *
 * Created on 16. května 2016, 15:08
 */

#ifndef TNLGRIDTRAVERSERCUDA_IMPL_H
#define TNLGRIDTRAVERSERCUDA_IMPL_H

/****
 * 1D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities >
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
   
   if( coordinates.x() <= end->x() )
   {
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {
         entity.refresh();
         EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
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
      bool processOnlyBoundaryEntities >
void
tnlGridTraverser< tnlGrid< 1, Real, tnlCuda, Index > >::
processEntities(
   const GridType& grid,
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
   typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
      
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );

   for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
      tnlGridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities >
         <<< cudaBlocks, cudaBlockSize >>>
         ( kernelGrid,
           kernelUserData,
           kernelBegin,
           kernelEnd,
           kernelEntityOrientation,
           kernelEntityBasis,
           gridXIdx );
   cudaThreadSynchronize();
   checkCudaDevice;
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
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
   const GridType& grid,
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
   typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
      
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );

   for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         tnlGridTraverser2D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              kernelBegin,
              kernelEnd,
              kernelEntityOrientation,
              kernelEntityBasis,
              gridXIdx,
              gridYIdx );
   cudaThreadSynchronize();
   checkCudaDevice;   
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
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
   const GridType& grid,
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
   typename GridEntity::MeshType* kernelGrid = tnlCuda::passToDevice( grid );
   UserData* kernelUserData = tnlCuda::passToDevice( userData );
      
   dim3 cudaBlockSize( 8, 8, 8 );
   dim3 cudaBlocks;
   cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   cudaBlocks.z = tnlCuda::getNumberOfBlocks( end.z() - begin.z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = tnlCuda::getNumberOfGrids( cudaBlocks.z );

   for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
            tnlGridTraverser3D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities >
               <<< cudaBlocks, cudaBlockSize >>>
               ( kernelGrid,
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
   tnlCuda::freeFromDevice( kernelGrid );
   tnlCuda::freeFromDevice( kernelBegin );
   tnlCuda::freeFromDevice( kernelEnd );
   tnlCuda::freeFromDevice( kernelEntityOrientation );
   tnlCuda::freeFromDevice( kernelEntityBasis );
   tnlCuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}

#endif /* TNLGRIDTRAVERSERCUDA_IMPL_H */

