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
#include <TNL/Matrices/MatrixPermutationApplier.h>

namespace TNL {
namespace Meshes {

template< typename Mesh, int Dimension >
struct IndexPermutationApplier
{
private:
   using GlobalIndexArray = typename Mesh::GlobalIndexArray;

   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                Subdimension >::storageEnabled
             >
   struct _SubentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& perm )
      {
         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Dimension, Subdimension >();
         Matrices::permuteMatrixRows( subentitiesStorage, perm );
      }
   };

   template< int Subdimension >
   struct _SubentitiesStorageWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                  Superdimension >::storageEnabled
             >
   struct _SuperentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& perm )
      {
         permuteArray( mesh.template getSuperentitiesCountsArray< Dimension, Superdimension >(), perm );
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Dimension, Superdimension >();
         Matrices::permuteMatrixRows( superentitiesStorage, perm );
      }
   };

   template< int Superdimension >
   struct _SuperentitiesStorageWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Subdimension >::EntityTopology,
                                                                  Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSubentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm )
      {
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Subdimension, Dimension >();
         Matrices::permuteMatrixColumns( superentitiesStorage, iperm );
      }
   };

   template< int Subdimension >
   struct IndexPermutationApplierSubentitiesWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Superdimension >::EntityTopology,
                                                                Dimension >::storageEnabled
             >
   struct IndexPermutationApplierSuperentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm )
      {
         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Superdimension, Dimension >();
         Matrices::permuteMatrixColumns( subentitiesStorage, iperm );
      }
   };

   template< int Superdimension >
   struct IndexPermutationApplierSuperentitiesWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
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

   template< typename Mesh_, std::enable_if_t< Mesh_::Config::dualGraphStorage(), bool > = true >
   static void permuteDualGraph( Mesh_& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm )
   {
      permuteArray( mesh.getNeighborCounts(), perm );
      auto& graph = mesh.getDualGraph();
      Matrices::permuteMatrixRows( graph, perm );
      Matrices::permuteMatrixColumns( graph, iperm );
   }

   template< typename Mesh_, std::enable_if_t< ! Mesh_::Config::dualGraphStorage(), bool > = true >
   static void permuteDualGraph( Mesh_& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm ) {}

public:
   template< typename Array >
   static void permuteArray( Array& array, const GlobalIndexArray& perm )
   {
      using IndexType = typename Array::IndexType;
      using DeviceType = typename Array::DeviceType;

      Array buffer( array.getSize() );

      // kernel to copy entities to new array, applying the permutation
      auto kernel1 = [] __cuda_callable__
         ( IndexType i,
           const typename Array::ValueType* array,
           typename Array::ValueType* buffer,
           const IndexType* perm )
      {
         buffer[ i ] = array[ perm[ i ] ];
      };

      // kernel to copy permuted entities back to the mesh
      auto kernel2 = [] __cuda_callable__
         ( IndexType i,
           typename Array::ValueType* array,
           const typename Array::ValueType* buffer )
      {
         array[ i ] = buffer[ i ];
      };

      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, array.getSize(),
                                                   kernel1,
                                                   array.getData(),
                                                   buffer.getData(),
                                                   perm.getData() );
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, array.getSize(),
                                                   kernel2,
                                                   array.getData(),
                                                   buffer.getData() );
   }

   static void exec( Mesh& mesh,
                     const GlobalIndexArray& perm,
                     const GlobalIndexArray& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      using DeviceType = typename Mesh::DeviceType;

      if( Dimension == 0 )
         permuteArray( mesh.getPoints(), perm );

      // permute superentities storage
      Algorithms::TemplateStaticFor< int, 0, Dimension, SubentitiesStorageWorker >::execHost( mesh, perm );

      // permute subentities storage
      Algorithms::TemplateStaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesStorageWorker >::execHost( mesh, perm );

      // update superentity indices from the subentities
      Algorithms::TemplateStaticFor< int, 0, Dimension, SubentitiesWorker >::execHost( mesh, iperm );

      // update subentity indices from the superentities
      Algorithms::TemplateStaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, SuperentitiesWorker >::execHost( mesh, iperm );

      if( Dimension == Mesh::getMeshDimension() ) {
         // permute dual graph
         permuteDualGraph( mesh, perm, iperm );
      }
   }
};

} // namespace Meshes
} // namespace TNL
