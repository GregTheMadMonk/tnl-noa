/***************************************************************************
                          Traverser_impl.h  -  description
                             -------------------
    begin                : Dec 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Traverser.h>

#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Meshes {   

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processBoundaryEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getBoundaryEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) i = 0; i < entitiesCount; i++ ) {
      const auto entityIndex = meshPointer->template getBoundaryEntityIndex< EntitiesDimension >( i );
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processInteriorEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getInteriorEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) i = 0; i < entitiesCount; i++ ) {
      const auto entityIndex = meshPointer->template getInteriorEntityIndex< EntitiesDimension >( i );
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processAllEntities( const MeshPointer& meshPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   auto entitiesCount = meshPointer->template getEntitiesCount< EntitiesDimension >();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
   for( decltype(entitiesCount) entityIndex = 0; entityIndex < entitiesCount; entityIndex++ ) {
      auto& entity = meshPointer->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *meshPointer, *userDataPointer, entity );
   }
}


#ifdef HAVE_CUDA
template< int EntitiesDimension,
          typename EntitiesProcessor,
          typename Mesh,
          typename UserData >
__global__ void
MeshTraverserBoundaryEntitiesKernel( const Mesh* mesh,
                                     UserData* userData,
                                     typename Mesh::GlobalIndexType entitiesCount )
{
   for( typename Mesh::GlobalIndexType i = blockIdx.x * blockDim.x + threadIdx.x;
        i < entitiesCount;
        i += blockDim.x * gridDim.x )
   {
      const auto entityIndex = mesh->template getBoundaryEntityIndex< EntitiesDimension >( i );
      auto& entity = mesh->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *mesh, *userData, entity );
   }
}

template< int EntitiesDimension,
          typename EntitiesProcessor,
          typename Mesh,
          typename UserData >
__global__ void
MeshTraverserInteriorEntitiesKernel( const Mesh* mesh,
                                     UserData* userData,
                                     typename Mesh::GlobalIndexType entitiesCount )
{
   for( typename Mesh::GlobalIndexType i = blockIdx.x * blockDim.x + threadIdx.x;
        i < entitiesCount;
        i += blockDim.x * gridDim.x )
   {
      const auto entityIndex = mesh->template getInteriorEntityIndex< EntitiesDimension >( i );
      auto& entity = mesh->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *mesh, *userData, entity );
   }
}

template< int EntitiesDimension,
          typename EntitiesProcessor,
          typename Mesh,
          typename UserData >
__global__ void
MeshTraverserAllEntitiesKernel( const Mesh* mesh,
                                UserData* userData,
                                typename Mesh::GlobalIndexType entitiesCount )
{
   for( typename Mesh::GlobalIndexType entityIndex = blockIdx.x * blockDim.x + threadIdx.x;
        entityIndex < entitiesCount;
        entityIndex += blockDim.x * gridDim.x )
   {
      auto& entity = mesh->template getEntity< EntitiesDimension >( entityIndex );
      // TODO: if the Mesh::IdType is void, then we should also pass the entityIndex
      EntitiesProcessor::processEntity( *mesh, *userData, entity );
   }
}

#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Traverser_minBlocksPerMultiprocessor = 8;
#else
   static constexpr int Traverser_minBlocksPerMultiprocessor = 4;
#endif
#endif

template< typename MeshConfig,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh< MeshConfig, Devices::Cuda >, MeshEntity, EntitiesDimension >::
processBoundaryEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
#ifdef HAVE_CUDA
   auto entitiesCount = meshPointer->template getBoundaryEntitiesCount< EntitiesDimension >();

   dim3 blockSize( 256 );
   dim3 gridSize;
   const int desGridSize = 4 * Traverser_minBlocksPerMultiprocessor
                             * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( entitiesCount, blockSize.x ) );

   Devices::Cuda::synchronizeDevice();
   MeshTraverserBoundaryEntitiesKernel< EntitiesDimension, EntitiesProcessor >
      <<< gridSize, blockSize >>>
      ( &meshPointer.template getData< Devices::Cuda >(),
        &userDataPointer.template modifyData< Devices::Cuda >(),
        entitiesCount );
   cudaDeviceSynchronize();
   checkCudaDevice;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename MeshConfig,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh< MeshConfig, Devices::Cuda >, MeshEntity, EntitiesDimension >::
processInteriorEntities( const MeshPointer& meshPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
#ifdef HAVE_CUDA
   auto entitiesCount = meshPointer->template getInteriorEntitiesCount< EntitiesDimension >();

   dim3 blockSize( 256 );
   dim3 gridSize;
   const int desGridSize = 4 * Traverser_minBlocksPerMultiprocessor
                             * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( entitiesCount, blockSize.x ) );

   Devices::Cuda::synchronizeDevice();
   MeshTraverserInteriorEntitiesKernel< EntitiesDimension, EntitiesProcessor >
      <<< gridSize, blockSize >>>
      ( &meshPointer.template getData< Devices::Cuda >(),
        &userDataPointer.template modifyData< Devices::Cuda >(),
        entitiesCount );
   cudaDeviceSynchronize();
   checkCudaDevice;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename MeshConfig,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Mesh< MeshConfig, Devices::Cuda >, MeshEntity, EntitiesDimension >::
processAllEntities( const MeshPointer& meshPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
#ifdef HAVE_CUDA
   auto entitiesCount = meshPointer->template getEntitiesCount< EntitiesDimension >();

   dim3 blockSize( 256 );
   dim3 gridSize;
   const int desGridSize = 4 * Traverser_minBlocksPerMultiprocessor
                             * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
   gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( entitiesCount, blockSize.x ) );

   Devices::Cuda::synchronizeDevice();
   MeshTraverserAllEntitiesKernel< EntitiesDimension, EntitiesProcessor >
      <<< gridSize, blockSize >>>
      ( &meshPointer.template getData< Devices::Cuda >(),
        &userDataPointer.template modifyData< Devices::Cuda >(),
        entitiesCount );
   cudaDeviceSynchronize();
   checkCudaDevice;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Meshes
} // namespace TNL
