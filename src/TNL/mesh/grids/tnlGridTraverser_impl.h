/***************************************************************************
                          tnlGridTraverser_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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
tnlGridTraverser< tnlGrid< 1, Real, Devices::Host, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{

 
   GridEntity entity( grid );
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
   const tnlGrid< 1, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
 
   coordinates.x() = begin->x() + ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
 
   GridEntity entity( *grid, coordinates, *entityOrientation, *entityBasis );
 
   if( coordinates.x() <= end->x() )
   {
      //if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {
         entity.refresh();
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
   const tnlGrid< 1, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, Devices::Cuda, Index > GridType;
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
tnlGridTraverser< tnlGrid< 1, Real, Devices::Cuda, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
#ifdef HAVE_CUDA
   CoordinatesType* kernelBegin = Devices::Cuda::passToDevice( begin );
   CoordinatesType* kernelEnd = Devices::Cuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = Devices::Cuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = Devices::Cuda::passToDevice( entityBasis );
   typename GridEntity::MeshType* kernelGrid = Devices::Cuda::passToDevice( grid );
   UserData* kernelUserData = Devices::Cuda::passToDevice( userData );
 
   if( processOnlyBoundaryEntities )
   {
      dim3 cudaBlockSize( 2 );
      dim3 cudaBlocks( 1 );
      tnlGridBoundaryTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
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
      cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
      const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );

      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         tnlGridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize >>>
            ( kernelGrid,
              kernelUserData,
              kernelBegin,
              kernelEnd,
              kernelEntityOrientation,
              kernelEntityBasis,
              gridXIdx );
   }
   cudaThreadSynchronize();
   checkCudaDevice;
   Devices::Cuda::freeFromDevice( kernelGrid );
   Devices::Cuda::freeFromDevice( kernelBegin );
   Devices::Cuda::freeFromDevice( kernelEnd );
   Devices::Cuda::freeFromDevice( kernelEntityOrientation );
   Devices::Cuda::freeFromDevice( kernelEntityBasis );
   Devices::Cuda::freeFromDevice( kernelUserData );
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
tnlGridTraverser< tnlGrid< 2, Real, Devices::Host, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType begin,
   const CoordinatesType end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
   GridEntity entity( grid );
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
//#pragma omp parallel for firstprivate( entity, begin, end ) if( Devices::Host::isOMPEnabled() )
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
   const tnlGrid< 2, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,
   const Index gridXIdx,
   const Index gridYIdx )
{
   typedef tnlGrid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin->x() + ( gridXIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin->y() + ( gridYIdx * Devices::Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
 
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
tnlGridTraverser< tnlGrid< 2, Real, Devices::Cuda, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
#ifdef HAVE_CUDA
   CoordinatesType* kernelBegin = Devices::Cuda::passToDevice( begin );
   CoordinatesType* kernelEnd = Devices::Cuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = Devices::Cuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = Devices::Cuda::passToDevice( entityBasis );
   typename GridEntity::MeshType* kernelGrid = Devices::Cuda::passToDevice( grid );
   UserData* kernelUserData = Devices::Cuda::passToDevice( userData );
 
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = Devices::Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.y );

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
   Devices::Cuda::freeFromDevice( kernelGrid );
   Devices::Cuda::freeFromDevice( kernelBegin );
   Devices::Cuda::freeFromDevice( kernelEnd );
   Devices::Cuda::freeFromDevice( kernelEntityOrientation );
   Devices::Cuda::freeFromDevice( kernelEntityBasis );
   Devices::Cuda::freeFromDevice( kernelUserData );
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
tnlGridTraverser< tnlGrid< 3, Real, Devices::Host, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
   GridEntity entity( grid );
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
   const tnlGrid< 3, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType* begin,
   const typename GridEntity::CoordinatesType* end,
   const typename GridEntity::CoordinatesType* entityOrientation,
   const typename GridEntity::CoordinatesType* entityBasis,
   const Index gridXIdx,
   const Index gridYIdx,
   const Index gridZIdx )
{
   typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin->x() + ( gridXIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin->y() + ( gridYIdx * Devices::Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = begin->z() + ( gridZIdx * Devices::Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
 
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
tnlGridTraverser< tnlGrid< 3, Real, Devices::Cuda, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,
   UserData& userData )
{
#ifdef HAVE_CUDA
   CoordinatesType* kernelBegin = Devices::Cuda::passToDevice( begin );
   CoordinatesType* kernelEnd = Devices::Cuda::passToDevice( end );
   CoordinatesType* kernelEntityOrientation = Devices::Cuda::passToDevice( entityOrientation );
   CoordinatesType* kernelEntityBasis = Devices::Cuda::passToDevice( entityBasis );
   typename GridEntity::MeshType* kernelGrid = Devices::Cuda::passToDevice( grid );
   UserData* kernelUserData = Devices::Cuda::passToDevice( userData );
 
   dim3 cudaBlockSize( 8, 8, 8 );
   dim3 cudaBlocks;
   cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = Devices::Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   cudaBlocks.z = Devices::Cuda::getNumberOfBlocks( end.z() - begin.z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.z );

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
   Devices::Cuda::freeFromDevice( kernelGrid );
   Devices::Cuda::freeFromDevice( kernelBegin );
   Devices::Cuda::freeFromDevice( kernelEnd );
   Devices::Cuda::freeFromDevice( kernelEntityOrientation );
   Devices::Cuda::freeFromDevice( kernelEntityBasis );
   Devices::Cuda::freeFromDevice( kernelUserData );
   checkCudaDevice;
#endif
}

} // namespace TNL

