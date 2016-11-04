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
 *   for( int dimensions = 0; dimensions < MeshTraitsType::meshDimensions; dimensions++ )
 *      for( int superdimensions = dimensions + 1; superdimensions <= MeshTraitsType::meshDimensions; superdimensions++ )
 *         if( EntityTraits< dimensions >::SuperentityTraits< superdimensions >::storageEnabled )
 *            for( GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< dimensions >(); i++ )
 *            {
 *               auto& entity = mesh.template getEntity< dimensions >( i );
 *               entity.template bindSuperentitiesStorageNetwork< superdimensions >( mesh.template getSuperentityStorageNetwork< superdimensions >().getValues( i ) );
 *            }
 */

#include <TNL/Meshes/MeshDimensionsTag.h>

namespace TNL {
namespace Meshes {

template< typename Mesh,
          typename DimensionsTag,
          typename SuperdimensionsTag,
          bool Enabled =
             Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< DimensionsTag::value >::EntityTopology,
                                                               SuperdimensionsTag::value >::storageEnabled
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
          typename DimensionsTag,
          typename SuperdimensionsTag >
struct MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionsTag, SuperdimensionsTag, false >
{
   template< typename Worker >
   static void exec( Mesh& mesh ) {}
};


template< typename Mesh,
          typename DimensionsTag,
          typename SuperdimensionsTag,
          bool Enabled =
             Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< SuperdimensionsTag::value >::EntityTopology,
                                                             DimensionsTag::value >::storageEnabled
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
          typename DimensionsTag,
          typename SuperdimensionsTag >
struct MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionsTag, SuperdimensionsTag, false >
{
   template< typename Worker >
   static void exec( Mesh& mesh ) {}
};


template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
struct MeshEntityStorageRebinderWorker
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderSuperentityWorker< Mesh, DimensionsTag, SuperdimensionsTag >::
         template exec< MeshEntityStorageRebinderWorker >( mesh );
      MeshEntityStorageRebinderSubentityWorker< Mesh, DimensionsTag, SuperdimensionsTag >::
         template exec< MeshEntityStorageRebinderWorker >( mesh );
   }

   static void bindSuperentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< DimensionsTag::value >(); i++ )
      {
         auto& subentity = mesh.template getEntity< DimensionsTag::value >( i );
         auto& superentitiesStorage = mesh.template getSuperentityStorageNetwork< DimensionsTag::value, SuperdimensionsTag::value >();
         subentity.template bindSuperentitiesStorageNetwork< SuperdimensionsTag::value >( superentitiesStorage.getValues( i ) );
      }
   }

   static void bindSubentities( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< SuperdimensionsTag::value >(); i++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( i );
         auto& subentitiesStorage = mesh.template getSubentityStorageNetwork< SuperdimensionsTag::value, DimensionsTag::value >();
         superentity.template bindSubentitiesStorageNetwork< DimensionsTag::value >( subentitiesStorage.getValues( i ) );
      }
   }
};


template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
struct MeshEntityStorageRebinderInner
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderWorker< Mesh, DimensionsTag, SuperdimensionsTag >::exec( mesh );
      MeshEntityStorageRebinderInner< Mesh, DimensionsTag, typename SuperdimensionsTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh, typename SuperdimensionsTag >
struct MeshEntityStorageRebinderInner< Mesh, typename SuperdimensionsTag::Decrement, SuperdimensionsTag >
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderWorker< Mesh, typename SuperdimensionsTag::Decrement, SuperdimensionsTag >::exec( mesh );
   }
};


template< typename Mesh, typename DimensionsTag = typename MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions >::Decrement >
struct MeshEntityStorageRebinder
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderInner< Mesh, DimensionsTag, MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions > >::exec( mesh );
      MeshEntityStorageRebinder< Mesh, typename DimensionsTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh >
struct MeshEntityStorageRebinder< Mesh, MeshDimensionsTag< 0 > >
{
   static void exec( Mesh& mesh )
   {
      MeshEntityStorageRebinderInner< Mesh, MeshDimensionsTag< 0 >, MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions > >::exec( mesh );
   }
};

} // namespace Meshes
} // namespace TNL
