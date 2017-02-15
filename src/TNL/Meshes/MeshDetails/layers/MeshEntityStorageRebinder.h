/***************************************************************************
                          MeshEntityStorageRebinder.h  -  description
                             -------------------
    begin                : Oct 22, 2016
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

/*
 * Everything in this file is basically just a templatized version of the
 * following pseudo-code (which does not work because normal variables are not
 * usable in template arguments):
 *
 *   for( int dimension = 0; dimension < MeshTraitsType::meshDimension; dimension++ )
 *      for( int superdimension = dimension + 1; superdimension <= MeshTraitsType::meshDimension; superdimension++ )
 *         if( EntityTraits< dimension >::SuperentityTraits< superdimension >::storageEnabled )
 *            for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< dimension >(); i++ )
 *            {
 *               auto& entity = mesh.template getEntity< dimension >( i );
 *               entity.template bindSuperentitiesStorageNetwork< superdimension >( mesh.template getSuperentityStorageNetwork< superdimension >().getValues( i ) );
 *            }
 */

#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/DevicePointer.h>

namespace TNL {
namespace Meshes {

template< typename Mesh,
          typename DimensionTag,
          typename SuperdimensionTag,
          bool Enabled =
             Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< DimensionTag::value >::EntityTopology,
                                                               SuperdimensionTag::value >::storageEnabled
          >
struct MeshEntityStorageRebinderSuperentityWorker
{
   template< typename Worker >
   static void exec( Mesh& mesh )
   {
      // If the reader is wondering why the code is in the Worker and not here:
      // that's because we're accessing protected method bindSuperentitiesStorageNetwork
      // and friend templates in GCC 6.1 apparently don't play nice with partial
      // template specializations.
      Worker::bindSuperentities( mesh );
   }
};

template< typename Mesh,
          typename DimensionTag,
          typename SuperdimensionTag >
struct MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionTag, SuperdimensionTag, false >
{
   template< typename Worker >
   static void exec( Mesh& mesh ) {}
};


template< typename Mesh,
          typename DimensionTag,
          typename SuperdimensionTag,
          bool Enabled =
             Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< SuperdimensionTag::value >::EntityTopology,
                                                             DimensionTag::value >::storageEnabled
          >
struct MeshEntityStorageRebinderSubentityWorker
{
   template< typename Worker >
   static void exec( Mesh& mesh )
   {
      // If the reader is wondering why the code is in the Worker and not here:
      // that's because we're accessing protected method bindSubentitiesStorageNetwork
      // and friend templates in GCC 6.1 apparently don't play nice with partial
      // template specializations.
      Worker::bindSubentities( mesh );
   }
};

template< typename Mesh,
          typename DimensionTag,
          typename SuperdimensionTag >
struct MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionTag, SuperdimensionTag, false >
{
   template< typename Worker >
   static void exec( Mesh& mesh ) {}
};


// This is split from everything else to make the friend specification (to use bindSubentitiesStorageNetwork)
// as easy as possible.
template< typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderWorker
{
   template< typename Index, typename Entity, typename Multimap >
   __cuda_callable__
   static void bindSuperentity( Index i, Entity& subentity, Multimap& superentitiesStorage )
   {
      subentity.template bindSuperentitiesStorageNetwork< SuperdimensionTag::value >( superentitiesStorage.getValues( i ) );
   }

   template< typename Index, typename Entity, typename Multimap >
   __cuda_callable__
   static void bindSubentity( Index i, Entity& superentity, Multimap& subentitiesStorage )
   {
      superentity.template bindSubentitiesStorageNetwork< DimensionTag::value >( subentitiesStorage.getValues( i ) );
   }
};


template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderDivisor
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderDivisor >( mesh );
      MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderDivisor >( mesh );
   }

   static void bindSuperentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getEntitiesCount< DimensionTag::value >(); i++ )
      {
         auto& subentity = mesh.template getEntity< DimensionTag::value >( i );
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< DimensionTag::value, SuperdimensionTag::value >();
         MeshEntityStorageRebinderWorker< DimensionTag, SuperdimensionTag >::bindSuperentity( i, subentity, superentitiesStorage );
      }
   }

   static void bindSubentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getEntitiesCount< SuperdimensionTag::value >(); i++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( i );
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< SuperdimensionTag::value, DimensionTag::value >();
         MeshEntityStorageRebinderWorker< DimensionTag, SuperdimensionTag >::bindSubentity( i, superentity, subentitiesStorage );
      }
   }
};

#ifdef HAVE_CUDA
template< typename Mesh,
          typename Multimap,
          typename DimensionTag,
          typename SuperdimensionTag >
__global__ void
MeshSuperentityRebinderKernel( Mesh* mesh,
                               Multimap* superentitiesStorage,
                               typename Mesh::GlobalIndexType entitiesCount )
{
   for( typename Mesh::GlobalIndexType i = blockIdx.x * blockDim.x + threadIdx.x;
        i < entitiesCount;
        i += blockDim.x * gridDim.x )
   {
      auto& subentity = mesh->template getEntity< DimensionTag::value >( i );
      MeshEntityStorageRebinderWorker< DimensionTag, SuperdimensionTag >::bindSuperentity( i, subentity, *superentitiesStorage );
   }
}

template< typename Mesh,
          typename Multimap,
          typename DimensionTag,
          typename SuperdimensionTag >
__global__ void
MeshSubentityRebinderKernel( Mesh* mesh,
                             Multimap* subentitiesStorage,
                             typename Mesh::GlobalIndexType entitiesCount )
{
   for( typename Mesh::GlobalIndexType i = blockIdx.x * blockDim.x + threadIdx.x;
        i < entitiesCount;
        i += blockDim.x * gridDim.x )
   {
      auto& superentity = mesh->template getEntity< SuperdimensionTag::value >( i );
      MeshEntityStorageRebinderWorker< DimensionTag, SuperdimensionTag >::bindSubentity( i, superentity, *subentitiesStorage );
   }
}
#endif

template< typename MeshConfig, typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderDivisor< Meshes::Mesh< MeshConfig, Devices::Cuda >, DimensionTag, SuperdimensionTag >
{
   using Mesh = Meshes::Mesh< MeshConfig, Devices::Cuda >;

   static void exec( Mesh& mesh )
   {
#ifdef HAVE_CUDA
      MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderDivisor >( mesh );
      MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderDivisor >( mesh );
#endif
   }

#ifdef HAVE_CUDA
   #if (__CUDA_ARCH__ >= 300 )
      static constexpr int minBlocksPerMultiprocessor = 8;
   #else
      static constexpr int minBlocksPerMultiprocessor = 4;
   #endif

   static void bindSuperentities( Mesh& mesh )
   {
      const auto entitiesCount = mesh.template getEntitiesCount< DimensionTag::value >();
      auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< DimensionTag::value, SuperdimensionTag::value >();
      using Multimap = typename std::remove_reference< decltype(superentitiesStorage) >::type;
      DevicePointer< Mesh > meshPointer( mesh );
      DevicePointer< Multimap > superentitiesStoragePointer( superentitiesStorage );
      Devices::Cuda::synchronizeDevice();

      dim3 blockSize( 256 );
      dim3 gridSize;
      const int desGridSize = 4 * minBlocksPerMultiprocessor
                                * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( entitiesCount, blockSize.x ) );

      MeshSuperentityRebinderKernel< Mesh, Multimap, DimensionTag, SuperdimensionTag >
         <<< gridSize, blockSize >>>
         ( &meshPointer.template modifyData< Devices::Cuda >(),
           &superentitiesStoragePointer.template modifyData< Devices::Cuda >(),
           entitiesCount );
   }

   static void bindSubentities( Mesh& mesh )
   {
      const auto entitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();
      auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< SuperdimensionTag::value, DimensionTag::value >();
      using Multimap = typename std::remove_reference< decltype(subentitiesStorage) >::type;
      DevicePointer< Mesh > meshPointer( mesh );
      DevicePointer< Multimap > subentitiesStoragePointer( subentitiesStorage );
      Devices::Cuda::synchronizeDevice();

      dim3 blockSize( 256 );
      dim3 gridSize;
      const int desGridSize = 4 * minBlocksPerMultiprocessor
                                * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( entitiesCount, blockSize.x ) );

      MeshSubentityRebinderKernel< Mesh, Multimap, DimensionTag, SuperdimensionTag >
         <<< gridSize, blockSize >>>
         ( &meshPointer.template modifyData< Devices::Cuda >(),
           &subentitiesStoragePointer.template modifyData< Devices::Cuda >(),
           entitiesCount );
   }
#endif
};


template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderInner
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderDivisor< Mesh, DimensionTag, SuperdimensionTag >::exec( mesh );
      MeshEntityStorageRebinderInner< Mesh, DimensionTag, typename SuperdimensionTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh, typename SuperdimensionTag >
struct MeshEntityStorageRebinderInner< Mesh, typename SuperdimensionTag::Decrement, SuperdimensionTag >
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderDivisor< Mesh, typename SuperdimensionTag::Decrement, SuperdimensionTag >::exec( mesh );
   }
};


template< typename Mesh, typename DimensionTag = typename Meshes::DimensionTag< Mesh::MeshTraitsType::meshDimension >::Decrement >
struct MeshEntityStorageRebinder
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderInner< Mesh, DimensionTag, Meshes::DimensionTag< Mesh::MeshTraitsType::meshDimension > >::exec( mesh );
      MeshEntityStorageRebinder< Mesh, typename DimensionTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh >
struct MeshEntityStorageRebinder< Mesh, Meshes::DimensionTag< 0 > >
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderInner< Mesh, Meshes::DimensionTag< 0 >, Meshes::DimensionTag< Mesh::MeshTraitsType::meshDimension > >::exec( mesh );
   }
};

} // namespace Meshes
} // namespace TNL
