/***************************************************************************
                          IndexPermutationApplier.h  -  description
                             -------------------
    begin                : Mar 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Experimental/Multimaps/MultimapPermutationApplier.h>

namespace TNL {
namespace Meshes {

template< typename Mesh, int Dimension >
struct IndexPermutationApplier
{
private:
   using IndexPermutationVector = typename Mesh::IndexPermutationVector;

   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                Subdimension >::storageEnabled
             >
   struct _SubentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& perm, bool& status )
      {
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< Dimension, Subdimension >();
         status &= permuteMultimapKeys( subentitiesStorage, perm );
      }
   };

   template< int Subdimension >
   struct _SubentitiesStorageWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                  Superdimension >::storageEnabled
             >
   struct _SuperentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& perm, bool& status )
      {
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< Dimension, Superdimension >();
         status &= permuteMultimapKeys( superentitiesStorage, perm );
      }
   };

   template< int Superdimension >
   struct _SuperentitiesStorageWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status ) {}
   };


   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Subdimension >::EntityTopology,
                                                                  Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSubentitiesWorker
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status )
      {
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< Subdimension, Dimension >();
         status &= permuteMultimapValues( superentitiesStorage, iperm );
      }
   };

   template< int Subdimension >
   struct IndexPermutationApplierSubentitiesWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Superdimension >::EntityTopology,
                                                                Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSuperentitiesWorker
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status )
      {
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< Superdimension, Dimension >();
         status &= permuteMultimapValues( subentitiesStorage, iperm );
      }
   };

   template< int Superdimension >
   struct IndexPermutationApplierSuperentitiesWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const IndexPermutationVector& iperm, bool& status ) {}
   };


   // template aliases needed to hide the 'Enabled' parameter
   template< int Subdimension >
   using SubentitiesStorageWorker = _SubentitiesStorageWorker< Subdimension >;

   template< int Superdimension >
   using SuperentitiesStorageWorker = _SuperentitiesStorageWorker< Superdimension >;

   template< int Subdimension >
   using SubentitiesWorker = IndexPermutationApplierSubentitiesWorker< Subdimension >;

   template< int Superdimension >
   using SuperentitiesWorker = IndexPermutationApplierSuperentitiesWorker< Superdimension >;

public:
   static bool exec( Mesh& mesh,
                     const IndexPermutationVector& perm,
                     const IndexPermutationVector& iperm )
   {
      const auto entitiesCount = mesh.template getEntitiesCount< Dimension >();

      using DeviceType = typename Mesh::DeviceType;
      using StorageArrayType = typename Mesh::template EntityTraits< Dimension >::StorageArrayType;
      StorageArrayType entities;
      if( ! entities.setSize( entitiesCount ) )
         return false;

      // kernel to copy entities to new array, applying the permutation
      auto kernel1 = [] __cuda_callable__
         ( typename Mesh::GlobalIndexType i,
           const Mesh* mesh,
           typename StorageArrayType::ElementType* entitiesArray,
           const typename Mesh::GlobalIndexType* perm )
      {
         entitiesArray[ i ] = mesh->template getEntity< Dimension >( perm[ i ] );
      };

      // kernel to copy permuted entities back to the mesh
      auto kernel2 = [] __cuda_callable__
         ( typename Mesh::GlobalIndexType i,
           Mesh* mesh,
           const typename StorageArrayType::ElementType* entitiesArray )
      {
         auto& entity = mesh->template getEntity< Dimension >( i );
         entity = entitiesArray[ i ];
         entity.setIndex( i );
      };

      DevicePointer< Mesh > meshPointer( mesh );
      ParallelFor< DeviceType >::exec( 0, entitiesCount,
                                       kernel1,
                                       &meshPointer.template getData< DeviceType >(),
                                       entities.getData(),
                                       perm.getData() );
      ParallelFor< DeviceType >::exec( 0, entitiesCount,
                                       kernel2,
                                       &meshPointer.template modifyData< DeviceType >(),
                                       entities.getData() );

      // hack due to StaticFor operating only with void return type
      bool status = true;

      // permute superentities storage
      StaticFor< int, 0, Dimension, SubentitiesStorageWorker >::exec( mesh, perm, status );

      // permute subentities storage
      StaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesStorageWorker >::exec( mesh, perm, status );

      // update superentity indices from the subentities
      StaticFor< int, 0, Dimension, SubentitiesWorker >::exec( mesh, iperm, status );

      // update subentity indices from the superentities
      StaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesWorker >::exec( mesh, iperm, status );

      return status;
   }
};

} // namespace Meshes
} // namespace TNL
