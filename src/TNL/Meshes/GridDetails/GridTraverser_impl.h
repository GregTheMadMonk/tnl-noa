/***************************************************************************
                          GridTraverser_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/MIC.h>

#pragma once

namespace TNL {
namespace Meshes {

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
GridTraverser< Meshes::Grid< 1, Real, Devices::Host, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream )
{
   GridEntity entity( *gridPointer );
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer );

      entity.getCoordinates() = begin;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
      entity.getCoordinates() = end;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
   }
   else
   {
      //TODO: This does not work with gcc-5.4 and older, should work at gcc 6.x
/*#pragma omp parallel for firstprivate( entity, begin, end ) if( Devices::Host::isOMPEnabled() )
      for( entity.getCoordinates().x() = begin.x();
           entity.getCoordinates().x() <= end.x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
      }*/ 
#ifdef HAVE_OPENMP
#pragma omp parallel firstprivate( begin, end ) if( Devices::Host::isOMPEnabled() )
#endif
      {
         GridEntity entity( *gridPointer );
#ifdef HAVE_OPENMP
#pragma omp for 
#endif
         for( IndexType x = begin.x(); x <= end.x(); x ++ )
         {
            entity.getCoordinates().x() = x;
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
         }      
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
GridTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
 
   coordinates.x() = begin.x() + ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( coordinates <= end )
   {   
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void
GridBoundaryTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;
 
   if( threadIdx.x == 0 )
   {
      coordinates.x() = begin.x();
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), *userData, entity );
   }
   if( threadIdx.x == 1 )
   {
      coordinates.x() = end.x();
      GridEntity entity( *grid, coordinates );
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
GridTraverser< Meshes::Grid< 1, Real, Devices::Cuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream )
{
#ifdef HAVE_CUDA
   auto& pool = CudaStreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Devices::Cuda::synchronizeDevice();
   if( processOnlyBoundaryEntities )
   {
      dim3 cudaBlockSize( 2 );
      dim3 cudaBlocks( 1 );
      GridBoundaryTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              &userDataPointer.template modifyData< Devices::Cuda >(),
              begin,
              end );
   }
   else
   {
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
      const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );

      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         GridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              &userDataPointer.template modifyData< Devices::Cuda >(),
              begin,
              end,
              gridXIdx );
   }

   // only launches into the stream 0 are synchronized
   if( stream == 0 )
   {
      cudaStreamSynchronize( s );
      checkCudaDevice;
   }
#endif
}

/****
 * 1D traverser, MIC
 */

template< typename Real,
          typename Index >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
GridTraverser< Meshes::Grid< 1, Real, Devices::MIC, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream )
{
    std::cout << "Not Implemented yet Grid Traverser <1, Real, Device::MIC>" << std::endl;
/*
   auto& pool = CudaStreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Devices::Cuda::synchronizeDevice();
   if( processOnlyBoundaryEntities )
   {
      dim3 cudaBlockSize( 2 );
      dim3 cudaBlocks( 1 );
      GridBoundaryTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              &userDataPointer.template modifyData< Devices::Cuda >(),
              begin,
              end );
   }
   else
   {
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
      const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );

      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         GridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              &userDataPointer.template modifyData< Devices::Cuda >(),
              begin,
              end,
              gridXIdx );
   }

   // only launches into the stream 0 are synchronized
   if( stream == 0 )
   {
      cudaStreamSynchronize( s );
      checkCudaDevice;
   }
*/
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
      int YOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 2, Real, Devices::Host, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer, begin, gridEntityParameters... );
      
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y() = begin.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            entity.getCoordinates().y() = end.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
         }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
         {
            entity.getCoordinates().x() = begin.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            entity.getCoordinates().x() = end.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
         }
   }
   else
   {
      //TODO: This does not work with gcc-5.4 and older, should work at gcc 6.x
/*#pragma omp parallel for firstprivate( entity, begin, end ) if( Devices::Host::isOMPEnabled() )
      for( entity.getCoordinates().y() = begin.y();
           entity.getCoordinates().y() <= end.y();
           entity.getCoordinates().y() ++ )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
         }*/
#ifdef HAVE_OPENMP
#pragma omp parallel firstprivate( begin, end ) if( Devices::Host::isOMPEnabled() )
#endif
      {
         GridEntity entity( *gridPointer, begin, gridEntityParameters... );
#ifdef HAVE_OPENMP
#pragma omp for 
#endif
         for( IndexType y = begin.y(); y <= end.y(); y ++ )
            for( IndexType x = begin.x(); x <= end.x(); x ++ )
            {
               entity.getCoordinates().x() = x;
               entity.getCoordinates().y() = y;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            }      
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
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void 
GridTraverser2D(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridXIdx,
   const Index gridYIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridXIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * Devices::Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  

   if( coordinates <= end )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {
         EntitiesProcessor::processEntity
         ( entity.getMesh(),
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
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 2, Real, Devices::Cuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
#ifdef HAVE_CUDA   
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaBlocks;
   cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = Devices::Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.y );

   auto& pool = CudaStreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Devices::Cuda::synchronizeDevice();
   for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         GridTraverser2D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              &userDataPointer.template modifyData< Devices::Cuda >(),
              begin,
              end,
              gridXIdx,
              gridYIdx,
              gridEntityParameters... );
 
   // only launches into the stream 0 are synchronized
   if( stream == 0 )
   {
      cudaStreamSynchronize( s );
      checkCudaDevice;
   }
#endif
}

/****
 * 2D traverser, MIC
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
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 2, Real, Devices::MIC, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
        
    
#ifdef HAVE_MIC   
   Devices::MIC::synchronizeDevice();

    //TOHLE JE PRUSER -- nemim poslat vypustku -- 
    //GridEntity entity( gridPointer.template getData< Devices::MIC >(), begin, gridEntityParameters... );


    Devices::MICHider<const GridType> hMicGrid;
    hMicGrid.pointer=& gridPointer.template getData< Devices::MIC >();
    Devices::MICHider<UserData> hMicUserData;
    hMicUserData.pointer=& userDataPointer.template modifyData<Devices::MIC>();
    TNLMICSTRUCT(begin, const CoordinatesType);
    TNLMICSTRUCT(end, const CoordinatesType);

    #pragma offload target(mic) in(sbegin,send,hMicUserData,hMicGrid)  
    {
        
        #pragma omp parallel firstprivate( sbegin, send )
        {     
            TNLMICSTRUCTUSE(begin, const CoordinatesType);
            TNLMICSTRUCTUSE(end, const CoordinatesType);    
            GridEntity entity( *(hMicGrid.pointer), *(kernelbegin) );
          
            if( processOnlyBoundaryEntities )
             {      
               if( YOrthogonalBoundary )
                  #pragma omp for
                  for( auto k = kernelbegin->x();
                       k <= kernelend->x();
                       k ++ )
                  {
                     entity.getCoordinates().x() = k;
                     entity.getCoordinates().y() = kernelbegin->y();
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), *(hMicUserData.pointer), entity );
                     entity.getCoordinates().y() = kernelend->y();
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), *(hMicUserData.pointer), entity );
                  }
               if( XOrthogonalBoundary )
                  #pragma omp for
                  for( auto k = kernelbegin->y();
                       k <= kernelend->y();
                       k ++ )
                  {
                     entity.getCoordinates().y() = k;
                     entity.getCoordinates().x() = kernelbegin->x();
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), *(hMicUserData.pointer), entity );
                     entity.getCoordinates().x() = kernelend->x();
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), *(hMicUserData.pointer), entity );
                  }
             }
            else
            {
                  #pragma omp for
                  for( IndexType y = kernelbegin->y(); y <= kernelend->y(); y ++ )
                     for( IndexType x = kernelbegin->x(); x <= kernelend->x(); x ++ )
                     {
                        // std::cerr << x << "   " <<y << std::endl;
                        entity.getCoordinates().x() = x;
                        entity.getCoordinates().y() = y;
                        entity.refresh();
                        EntitiesProcessor::processEntity( entity.getMesh(), *(hMicUserData.pointer), entity );
                     }      
             }
        }
    }
      
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
      int ZOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 3, Real, Devices::Host, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer, begin, gridEntityParameters... );
      
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
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
               entity.getCoordinates().z() = end.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
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
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
               entity.getCoordinates().y() = end.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
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
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
               entity.getCoordinates().x() = end.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            }
   }
   else
   {
      // TODO: this does not work with gcc-5.4 and older, should work at gcc 6.x
/*#pragma omp parallel for firstprivate( entity, begin, end ) if( Devices::Host::isOMPEnabled() )      
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
               EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            }*/
#ifdef HAVE_OPENMP
#pragma omp parallel firstprivate( begin, end ) if( Devices::Host::isOMPEnabled() )
#endif
      {
         GridEntity entity( *gridPointer, begin, gridEntityParameters... );
#ifdef HAVE_OPENMP
#pragma omp for
#endif
         for( IndexType z = begin.z(); z <= end.z(); z ++ )
            for( IndexType y = begin.y(); y <= end.y(); y ++ )
               for( IndexType x = begin.x(); x <= end.x(); x ++ )
               {
                  entity.getCoordinates().x() = x;
                  entity.getCoordinates().y() = y;
                  entity.getCoordinates().z() = z;
                  entity.refresh();
                  EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
            }
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
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void
GridTraverser3D(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridXIdx,
   const Index gridYIdx,
   const Index gridZIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridXIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * Devices::Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   coordinates.z() = begin.z() + ( gridZIdx * Devices::Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( coordinates <= end )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
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
         int ZOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 3, Real, Devices::Cuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
#ifdef HAVE_CUDA   
   dim3 cudaBlockSize( 8, 8, 8 );
   dim3 cudaBlocks;
   cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = Devices::Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   cudaBlocks.z = Devices::Cuda::getNumberOfBlocks( end.z() - begin.z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.z );

   auto& pool = CudaStreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Devices::Cuda::synchronizeDevice();
   for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
            GridTraverser3D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaBlocks, cudaBlockSize, 0, s >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 &userDataPointer.template modifyData< Devices::Cuda >(),
                 begin,
                 end,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 gridEntityParameters... );

   // only launches into the stream 0 are synchronized
   if( stream == 0 )
   {
      cudaStreamSynchronize( s );
      checkCudaDevice;
   }
#endif
}

/****
 * 3D traverser, MIC
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
         int ZOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 3, Real, Devices::MIC, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   SharedPointer< UserData, DeviceType >& userDataPointer,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
    std::cout << "Not Implemented yet Grid Traverser <3, Real, Device::MIC>" << std::endl;
    
/* HAVE_CUDA   
   dim3 cudaBlockSize( 8, 8, 8 );
   dim3 cudaBlocks;
   cudaBlocks.x = Devices::Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
   cudaBlocks.y = Devices::Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
   cudaBlocks.z = Devices::Cuda::getNumberOfBlocks( end.z() - begin.z() + 1, cudaBlockSize.z );
   const IndexType cudaXGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.x );
   const IndexType cudaYGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.y );
   const IndexType cudaZGrids = Devices::Cuda::getNumberOfGrids( cudaBlocks.z );

   auto& pool = CudaStreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Devices::Cuda::synchronizeDevice();
   for( IndexType gridZIdx = 0; gridZIdx < cudaZGrids; gridZIdx ++ )
      for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
         for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
            GridTraverser3D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaBlocks, cudaBlockSize, 0, s >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 &userDataPointer.template modifyData< Devices::Cuda >(),
                 begin,
                 end,
                 gridXIdx,
                 gridYIdx,
                 gridZIdx,
                 gridEntityParameters... );

   // only launches into the stream 0 are synchronized
   if( stream == 0 )
   {
      cudaStreamSynchronize( s );
      checkCudaDevice;
   }
 */
}

} // namespace Meshes
} // namespace TNL
