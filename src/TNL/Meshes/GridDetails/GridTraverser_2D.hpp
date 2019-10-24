/***************************************************************************
                          GridTraverser_2D.hpp  -  description
                             -------------------
    begin                : Jan 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/CudaStreamPool.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Meshes/GridDetails/GridTraverser.h>

namespace TNL {
namespace Meshes {

//#define GRID_TRAVERSER_USE_STREAMS


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
   UserData& userData,
   GridTraverserMode mode,
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
#ifdef HAVE_OPENMP
      if( Devices::Host::isOMPEnabled() )
      {
#pragma omp parallel firstprivate( begin, end )
         {
            GridEntity entity( *gridPointer );
#pragma omp for
            // TODO: g++ 5.5 crashes when coding this loop without auxiliary x and y as bellow
            for( IndexType y = begin.y(); y <= end.y(); y ++ )
               for( IndexType x = begin.x(); x <= end.x(); x ++ )
               {
                  entity.getCoordinates().x() = x;
                  entity.getCoordinates().y() = y;
                  entity.refresh();
                  EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               }
         }
      }
      else
      {
         GridEntity entity( *gridPointer );
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
#else
      GridEntity entity( *gridPointer );
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
#endif
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
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + Devices::Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = begin.y() + Devices::Cuda::getGlobalThreadIdx_y( gridIdx );
   
   if( coordinates <= end )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {
         EntitiesProcessor::processEntity
         ( *grid,
           userData,
           entity );
      }
   }
}

// Boundary traverser using streams
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void 
GridTraverser2DBoundaryAlongX(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginX,
   const Index endX,
   const Index fixedY,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = beginX + Devices::Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = fixedY;
   
   if( coordinates.x() <= endX )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      EntitiesProcessor::processEntity
      ( *grid,
        userData,
        entity );
   }
}

// Boundary traverser using streams
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void 
GridTraverser2DBoundaryAlongY(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginY,
   const Index endY,
   const Index fixedX,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = fixedX;
   coordinates.y() = beginY + Devices::Cuda::getGlobalThreadIdx_x( gridIdx );
   
   if( coordinates.y() <= endY )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      EntitiesProcessor::processEntity
      ( *grid,
        userData,
        entity );
   }   
}


template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void 
GridTraverser2DBoundary(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginX,
   const Index endX,
   const Index beginY,
   const Index endY,
   const Index blocksPerFace,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   using GridType = Meshes::Grid< 2, Real, Devices::Cuda, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   
   const Index faceIdx = blockIdx.x / blocksPerFace;
   const Index faceBlockIdx = blockIdx.x % blocksPerFace;
   const Index threadId = faceBlockIdx * blockDim. x + threadIdx.x;
   if( faceIdx < 2 )
   {
      const Index entitiesAlongX = endX - beginX + 1;
      if( threadId < entitiesAlongX )
      {
         GridEntity entity( *grid, 
            CoordinatesType(  beginX + threadId, faceIdx == 0 ? beginY : endY ),
            gridEntityParameters... );
         //printf( "faceIdx %d Thread %d -> %d %d \n ", faceIdx, threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
         entity.refresh();
         EntitiesProcessor::processEntity( *grid, userData, entity );
      }
   }
   else
   {
      const Index entitiesAlongY = endY - beginY - 1;   
      if( threadId < entitiesAlongY )
      {
         GridEntity entity( *grid, 
            CoordinatesType(  faceIdx == 2 ? beginX : endX, beginY + threadId + 1  ),
            gridEntityParameters... );
         //printf( "faceIdx %d Thread %d -> %d %d \n ", faceIdx, threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
         entity.refresh();
         EntitiesProcessor::processEntity( *grid, userData, entity );
      }
   }
   
   
   
   /*const Index aux = max( entitiesAlongX, entitiesAlongY );
   const Index& warpSize = Devices::Cuda::getWarpSize();
   const Index threadsPerAxis = warpSize * ( aux / warpSize + ( aux % warpSize != 0 ) );
   
   Index threadId = Devices::Cuda::getGlobalThreadIdx_x( gridIdx );
   GridEntity entity( *grid, 
         CoordinatesType( 0, 0 ),
         gridEntityParameters... );
   CoordinatesType& coordinates = entity.getCoordinates();
   const Index axisIndex = threadId / threadsPerAxis;
   //printf( "axisIndex %d, threadId %d thradsPerAxis %d \n", axisIndex, threadId, threadsPerAxis );   
   threadId -= axisIndex * threadsPerAxis;
   switch( axisIndex )
   {
      case 1:
         coordinates = CoordinatesType( beginX + threadId, beginY );
         if( threadId < entitiesAlongX )
         {
            //printf( "X1: Thread %d -> %d %d \n ", threadId, coordinates.x(), coordinates.y() );
            entity.refresh();
            EntitiesProcessor::processEntity( *grid, userData, entity );
         }
         break;
      case 2:
         coordinates = CoordinatesType( beginX + threadId, endY );
         if( threadId < entitiesAlongX )
         {
            //printf( "X2: Thread %d -> %d %d \n ", threadId, coordinates.x(), coordinates.y() );
            entity.refresh();
            EntitiesProcessor::processEntity( *grid, userData, entity );
         }
         break;
      case 3:
         coordinates = CoordinatesType( beginX, beginY + threadId + 1 );
         if( threadId < entitiesAlongY )
         {
            //printf( "Y1: Thread %d -> %d %d \n ", threadId, coordinates.x(), coordinates.y() );
            entity.refresh();
            EntitiesProcessor::processEntity( *grid, userData, entity );
         }
         break;
      case 4:
         coordinates = CoordinatesType( endX, beginY + threadId + 1 );
         if( threadId < entitiesAlongY )
         {
            //printf( "Y2: Thread %d -> %d %d \n ", threadId, coordinates.x(), coordinates.y() );
            entity.refresh();
            EntitiesProcessor::processEntity( *grid, userData, entity );
         }
         break;
   }*/
   
   /*if( threadId < entitiesAlongX )
   {
      GridEntity entity( *grid, 
         CoordinatesType( beginX + threadId, beginY ),
         gridEntityParameters... );
      //printf( "X1: Thread %d -> %d %d x %d %d \n ", threadId, 
      //   entity.getCoordinates().x(), entity.getCoordinates().y(),
      //   grid->getDimensions().x(), grid->getDimensions().y() );
      entity.refresh();
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }
   else if( ( threadId -= entitiesAlongX ) < entitiesAlongX && threadId >= 0 )
   {
      GridEntity entity( *grid, 
         CoordinatesType( beginX + threadId, endY ),
         gridEntityParameters... );
      entity.refresh();
      //printf( "X2: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }
   else if( ( ( threadId -= entitiesAlongX ) < entitiesAlongY - 1 ) && threadId >= 0 )
   {
      GridEntity entity( *grid,
         CoordinatesType( beginX, beginY + threadId + 1 ),
      gridEntityParameters... );
      entity.refresh();
      //printf( "Y1: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );      
   }
   else if( ( ( threadId -= entitiesAlongY - 1 ) < entitiesAlongY - 1  ) && threadId >= 0 )
   {
      GridEntity entity( *grid,
         CoordinatesType( endX, beginY + threadId + 1 ),
      gridEntityParameters... );
      entity.refresh();
      //printf( "Y2: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }*/
}


#endif // HAVE_CUDA

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
   UserData& userData,
   GridTraverserMode mode,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
#ifdef HAVE_CUDA
   if( processOnlyBoundaryEntities && 
       ( GridEntity::getEntityDimension() == 2 || GridEntity::getEntityDimension() == 0 ) )
   {
#ifdef GRID_TRAVERSER_USE_STREAMS
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocksCountAlongX, cudaGridsCountAlongX,
           cudaBlocksCountAlongY, cudaGridsCountAlongY;
      Devices::Cuda::setupThreads( cudaBlockSize, cudaBlocksCountAlongX, cudaGridsCountAlongX, end.x() - begin.x() + 1 );
      Devices::Cuda::setupThreads( cudaBlockSize, cudaBlocksCountAlongY, cudaGridsCountAlongY, end.y() - begin.y() - 1 );
            
      auto& pool = CudaStreamPool::getInstance();
      Devices::Cuda::synchronizeDevice();
      
      const cudaStream_t& s1 = pool.getStream( stream );
      const cudaStream_t& s2 = pool.getStream( stream + 1 );
      dim3 gridIdx, cudaGridSize;
      for( gridIdx.x = 0; gridIdx.x < cudaGridsCountAlongX.x; gridIdx.x++ )
      {
         Devices::Cuda::setupGrid( cudaBlocksCountAlongX, cudaGridsCountAlongX, gridIdx, cudaGridSize );
         //Devices::Cuda::printThreadsSetup( cudaBlockSize, cudaBlocksCountAlongX, cudaGridSize, cudaGridsCountAlongX );
         GridTraverser2DBoundaryAlongX< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s1 >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.x(),
                 end.x(),
                 begin.y(),
                 gridIdx,
                 gridEntityParameters... );
         GridTraverser2DBoundaryAlongX< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s2 >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.x(),
                 end.x(),
                 end.y(),
                 gridIdx,
                 gridEntityParameters... );
      }
      const cudaStream_t& s3 = pool.getStream( stream + 2 );
      const cudaStream_t& s4 = pool.getStream( stream + 3 );
      for( gridIdx.x = 0; gridIdx.x < cudaGridsCountAlongY.x; gridIdx.x++ )
      {
         Devices::Cuda::setupGrid( cudaBlocksCountAlongY, cudaGridsCountAlongY, gridIdx, cudaGridSize );
         GridTraverser2DBoundaryAlongY< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s3 >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.y() + 1,
                 end.y() - 1,
                 begin.x(),
                 gridIdx,
                 gridEntityParameters... );
         GridTraverser2DBoundaryAlongY< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s4 >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.y() + 1,
                 end.y() - 1,
                 end.x(),
                 gridIdx,
                 gridEntityParameters... );
      }
      cudaStreamSynchronize( s1 );
      cudaStreamSynchronize( s2 );
      cudaStreamSynchronize( s3 );
      cudaStreamSynchronize( s4 );
#else // not defined GRID_TRAVERSER_USE_STREAMS
      dim3 cudaBlockSize( 256 );      
      dim3 cudaBlocksCount, cudaGridsCount;
      const IndexType entitiesAlongX = end.x() - begin.x() + 1;
      const IndexType entitiesAlongY = end.x() - begin.x() - 1;
      const IndexType maxFaceSize = max( entitiesAlongX, entitiesAlongY );
      const IndexType blocksPerFace = maxFaceSize / cudaBlockSize.x + ( maxFaceSize % cudaBlockSize.x != 0 );
      IndexType cudaThreadsCount = 4 * cudaBlockSize.x * blocksPerFace;
      Devices::Cuda::setupThreads( cudaBlockSize, cudaBlocksCount, cudaGridsCount, cudaThreadsCount );
      //std::cerr << "blocksPerFace = " << blocksPerFace << "Threads count = " << cudaThreadsCount 
      //          << "cudaBlockCount = " << cudaBlocksCount.x << std::endl;      
      dim3 gridIdx, cudaGridSize;
      Devices::Cuda::synchronizeDevice();
      for( gridIdx.x = 0; gridIdx.x < cudaGridsCount.x; gridIdx.x++ )
      {
         Devices::Cuda::setupGrid( cudaBlocksCount, cudaGridsCount, gridIdx, cudaGridSize );
         //Devices::Cuda::printThreadsSetup( cudaBlockSize, cudaBlocksCountAlongX, cudaGridSize, cudaGridsCountAlongX );
         GridTraverser2DBoundary< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.x(),
                 end.x(),
                 begin.y(),
                 end.y(),
                 blocksPerFace,
                 gridIdx,
                 gridEntityParameters... );
      }
#endif //GRID_TRAVERSER_USE_STREAMS
      //getchar();      
      TNL_CHECK_CUDA_DEVICE;      
   }
   else
   {
      dim3 cudaBlockSize( 16, 16 );
      dim3 cudaBlocksCount, cudaGridsCount;
      Devices::Cuda::setupThreads( cudaBlockSize, cudaBlocksCount, cudaGridsCount,
                                   end.x() - begin.x() + 1,
                                   end.y() - begin.y() + 1 );
      
      auto& pool = CudaStreamPool::getInstance();
      const cudaStream_t& s = pool.getStream( stream );

      Devices::Cuda::synchronizeDevice();
      dim3 gridIdx, cudaGridSize;
      for( gridIdx.y = 0; gridIdx.y < cudaGridsCount.y; gridIdx.y ++ )
         for( gridIdx.x = 0; gridIdx.x < cudaGridsCount.x; gridIdx.x ++ )
         {
            Devices::Cuda::setupGrid( cudaBlocksCount, cudaGridsCount, gridIdx, cudaGridSize );
	    //Devices::Cuda::printThreadsSetup( cudaBlockSize, cudaBlocksCount, cudaGridSize, cudaGridsCount );
            GridTraverser2D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin,
                 end,
                 gridIdx,
                 gridEntityParameters... );
         }

#ifdef NDEBUG
   if( mode == synchronousMode )
   {
      cudaStreamSynchronize( s );
      TNL_CHECK_CUDA_DEVICE;
   }
#else
   cudaStreamSynchronize( s );
   TNL_CHECK_CUDA_DEVICE;
#endif
   }

#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Meshes
} // namespace TNL
