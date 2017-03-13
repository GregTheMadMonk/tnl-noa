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
#include <TNL/ParallelFor.h>
#include <TNL/StaticFor.h>

namespace TNL {
namespace Meshes {

template< typename Mesh >
class MeshEntityStorageRebinder
{
   using IndexType = typename Mesh::GlobalIndexType;
   using DeviceType = typename Mesh::DeviceType;

// nvcc does not allow __cuda_callable__ lambdas inside private or protected sections
#ifdef __NVCC__
public:
#endif
   template< typename DimensionTag,
             typename SuperdimensionTag,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< DimensionTag::value >::EntityTopology,
                                                                  SuperdimensionTag::value >::storageEnabled
             >
   struct SuperentityWorker
   {
      static void bindSuperentities( Mesh& mesh )
      {
         const IndexType entitiesCount = mesh.template getEntitiesCount< DimensionTag::value >();
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< DimensionTag::value, SuperdimensionTag::value >();
         using Multimap = typename std::remove_reference< decltype(superentitiesStorage) >::type;
         DevicePointer< Mesh > meshPointer( mesh );
         DevicePointer< Multimap > superentitiesStoragePointer( superentitiesStorage );

         auto kernel = [] __cuda_callable__
            ( IndexType i,
              Mesh* mesh,
              Multimap* superentitiesStorage )
         {
            auto& subentity = mesh->template getEntity< DimensionTag::value >( i );
            subentity.template bindSuperentitiesStorageNetwork< SuperdimensionTag::value >( superentitiesStorage->getValues( i ) );
         };

         ParallelFor< DeviceType >::exec( (IndexType) 0, entitiesCount,
                                          kernel,
                                          &meshPointer.template modifyData< DeviceType >(),
                                          &superentitiesStoragePointer.template modifyData< DeviceType >() );
      }
   };

   template< typename DimensionTag,
             typename SuperdimensionTag >
   struct SuperentityWorker< DimensionTag, SuperdimensionTag, false >
   {
      static void bindSuperentities( Mesh& mesh ) {}
   };


   template< typename DimensionTag,
             typename SuperdimensionTag,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< SuperdimensionTag::value >::EntityTopology,
                                                                DimensionTag::value >::storageEnabled
             >
   struct SubentityWorker
   {
      static void bindSubentities( Mesh& mesh )
      {
         const IndexType entitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< SuperdimensionTag::value, DimensionTag::value >();
         using Multimap = typename std::remove_reference< decltype(subentitiesStorage) >::type;
         DevicePointer< Mesh > meshPointer( mesh );
         DevicePointer< Multimap > subentitiesStoragePointer( subentitiesStorage );

         auto kernel = [] __cuda_callable__
            ( IndexType i,
              Mesh* mesh,
              Multimap* subentitiesStorage )
         {
            auto& superentity = mesh->template getEntity< SuperdimensionTag::value >( i );
            superentity.template bindSubentitiesStorageNetwork< DimensionTag::value >( subentitiesStorage->getValues( i ) );
         };

         ParallelFor< DeviceType >::exec( (IndexType) 0, entitiesCount,
                                          kernel,
                                          &meshPointer.template modifyData< DeviceType >(),
                                          &subentitiesStoragePointer.template modifyData< DeviceType >() );
      }
   };

   template< typename DimensionTag,
             typename SuperdimensionTag >
   struct SubentityWorker< DimensionTag, SuperdimensionTag, false >
   {
      static void bindSubentities( Mesh& mesh ) {}
   };


   template< int Dimension, int Superdimension >
   struct InnerLoop
   {
      static void exec( Mesh& mesh )
      {
         using DimensionTag = Meshes::DimensionTag< Dimension >;
         using SuperdimensionTag = Meshes::DimensionTag< Superdimension >;
         SuperentityWorker< DimensionTag, SuperdimensionTag >::bindSuperentities( mesh );
         SubentityWorker< DimensionTag, SuperdimensionTag >::bindSubentities( mesh );
      }
   };

   template< int Dimension >
   struct OuterLoop
   {
      template< int Superdimension >
      using Inner = InnerLoop< Dimension, Superdimension >;

      static void exec( Mesh& mesh )
      {
         StaticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1, Inner >::exec( mesh );
      }
   };

public:
   static void exec( Mesh& mesh )
   {
      StaticFor< int, 0, Mesh::getMeshDimension() + 1, OuterLoop >::exec( mesh );
   }
};

} // namespace Meshes
} // namespace TNL
