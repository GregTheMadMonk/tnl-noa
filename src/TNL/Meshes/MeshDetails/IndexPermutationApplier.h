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
#include <TNL/Containers/Multimaps/MultimapPermutationApplier.h>

namespace TNL {
namespace Meshes {

template< typename Mesh, int Dimension >
struct IndexPermutationApplier
{
private:
   using GlobalIndexVector = typename Mesh::GlobalIndexVector;

   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                Subdimension >::storageEnabled
             >
   struct _SubentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& perm )
      {
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< Dimension, Subdimension >();
         Containers::Multimaps::permuteMultimapKeys( subentitiesStorage, perm );
      }
   };

   template< int Subdimension >
   struct _SubentitiesStorageWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                  Superdimension >::storageEnabled
             >
   struct _SuperentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& perm )
      {
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< Dimension, Superdimension >();
         Containers::Multimaps::permuteMultimapKeys( superentitiesStorage, perm );
      }
   };

   template< int Superdimension >
   struct _SuperentitiesStorageWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm ) {}
   };


   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Subdimension >::EntityTopology,
                                                                  Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSubentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm )
      {
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< Subdimension, Dimension >();
         Containers::Multimaps::permuteMultimapValues( superentitiesStorage, iperm );
      }
   };

   template< int Subdimension >
   struct IndexPermutationApplierSubentitiesWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Superdimension >::EntityTopology,
                                                                Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSuperentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm )
      {
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< Superdimension, Dimension >();
         Containers::Multimaps::permuteMultimapValues( subentitiesStorage, iperm );
      }
   };

   template< int Superdimension >
   struct IndexPermutationApplierSuperentitiesWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexVector& iperm ) {}
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
   static void permutePoints( Mesh& mesh,
                              const GlobalIndexVector& perm,
                              const GlobalIndexVector& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      using DeviceType = typename Mesh::DeviceType;
      using PointArrayType = typename Mesh::MeshTraitsType::PointArrayType;

      const IndexType pointsCount = mesh.template getEntitiesCount< 0 >();

      PointArrayType points;
      points.setSize( pointsCount );

      // kernel to copy entities to new array, applying the permutation
      auto kernel1 = [] __cuda_callable__
         ( IndexType i,
           const Mesh* mesh,
           typename PointArrayType::ValueType* pointsArray,
           const IndexType* perm )
      {
         pointsArray[ i ] = mesh->getPoint( perm[ i ] );
      };

      // kernel to copy permuted entities back to the mesh
      auto kernel2 = [] __cuda_callable__
         ( IndexType i,
           Mesh* mesh,
           const typename PointArrayType::ValueType* pointsArray )
      {
         mesh->getPoint( i ) = pointsArray[ i ];
      };

      Pointers::DevicePointer< Mesh > meshPointer( mesh );
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, pointsCount,
                                                   kernel1,
                                                   &meshPointer.template getData< DeviceType >(),
                                                   points.getData(),
                                                   perm.getData() );
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, pointsCount,
                                                   kernel2,
                                                   &meshPointer.template modifyData< DeviceType >(),
                                                   points.getData() );
   }

   static void exec( Mesh& mesh,
                     const GlobalIndexVector& perm,
                     const GlobalIndexVector& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      using DeviceType = typename Mesh::DeviceType;
      using StorageArrayType = typename Mesh::template EntityTraits< Dimension >::StorageArrayType;

      if( Dimension == 0 )
         permutePoints( mesh, perm, iperm );

      // permute superentities storage
      Algorithms::TemplateStaticFor< int, 0, Dimension, SubentitiesStorageWorker >::execHost( mesh, perm );

      // permute subentities storage
      Algorithms::TemplateStaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesStorageWorker >::execHost( mesh, perm );

      // update superentity indices from the subentities
      Algorithms::TemplateStaticFor< int, 0, Dimension, SubentitiesWorker >::execHost( mesh, iperm );

      // update subentity indices from the superentities
      Algorithms::TemplateStaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesWorker >::execHost( mesh, iperm );
   }
};

} // namespace Meshes
} // namespace TNL
