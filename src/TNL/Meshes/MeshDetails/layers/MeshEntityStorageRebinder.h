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
 *            for( GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< dimension >(); i++ )
 *            {
 *               auto& entity = mesh.template getEntity< dimension >( i );
 *               entity.template bindSuperentitiesStorageNetwork< superdimension >( mesh.template getSuperentityStorageNetwork< superdimension >().getValues( i ) );
 *            }
 */

#include <TNL/Meshes/DimensionTag.h>

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


template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderWorker
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderWorker >( mesh );
      MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionTag, SuperdimensionTag >::
         template exec< MeshEntityStorageRebinderWorker >( mesh );
   }

   static void bindSuperentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< DimensionTag::value >(); i++ )
      {
         auto& subentity = mesh.template getEntity< DimensionTag::value >( i );
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< DimensionTag::value, SuperdimensionTag::value >();
         subentity.template bindSuperentitiesStorageNetwork< SuperdimensionTag::value >( superentitiesStorage.getValues( i ) );
      }
   }

   static void bindSubentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< SuperdimensionTag::value >(); i++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( i );
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< SuperdimensionTag::value, DimensionTag::value >();
         superentity.template bindSubentitiesStorageNetwork< DimensionTag::value >( subentitiesStorage.getValues( i ) );
      }
   }
};


template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderInner
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderWorker< Mesh, DimensionTag, SuperdimensionTag >::exec( mesh );
      MeshEntityStorageRebinderInner< Mesh, DimensionTag, typename SuperdimensionTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh, typename SuperdimensionTag >
struct MeshEntityStorageRebinderInner< Mesh, typename SuperdimensionTag::Decrement, SuperdimensionTag >
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderWorker< Mesh, typename SuperdimensionTag::Decrement, SuperdimensionTag >::exec( mesh );
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
