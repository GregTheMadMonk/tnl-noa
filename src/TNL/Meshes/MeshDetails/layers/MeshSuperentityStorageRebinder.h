/***************************************************************************
                          MeshSuperentityStorageRebinder.h  -  description
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

template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
struct MeshSuperentityStorageRebinderWorker
{
   static void exec( Mesh& mesh )
   {
      for( typename Mesh::GlobalIndexType i = 0; i < mesh.template getNumberOfEntities< DimensionsTag::value >(); i++ )
      {
         auto& entity = mesh.template getEntity< DimensionsTag::value >( i );
         auto& storage = mesh.template getSuperentityStorageNetwork< typename Mesh::template EntityTraits< DimensionsTag::value >::EntityTopology, SuperdimensionsTag >();
         entity.template bindSuperentitiesStorageNetwork< SuperdimensionsTag::value >( storage.getValues( i ) );
      }
   }
};


template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
struct MeshSuperentityStorageRebinderInner
{
   static void exec( Mesh& mesh )
   {
      MeshSuperentityStorageRebinderWorker< Mesh, DimensionsTag, SuperdimensionsTag >::exec( mesh );
      MeshSuperentityStorageRebinderInner< Mesh, DimensionsTag, typename SuperdimensionsTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh, typename SuperdimensionsTag >
struct MeshSuperentityStorageRebinderInner< Mesh, typename SuperdimensionsTag::Decrement, SuperdimensionsTag >
{
   static void exec( Mesh& mesh )
   {
      MeshSuperentityStorageRebinderWorker< Mesh, typename SuperdimensionsTag::Decrement, SuperdimensionsTag >::exec( mesh );
   }
};


template< typename Mesh, typename DimensionsTag = typename MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions >::Decrement >
struct MeshSuperentityStorageRebinder
{
   static void exec( Mesh& mesh )
   {
      MeshSuperentityStorageRebinderInner< Mesh, DimensionsTag, MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions > >::exec( mesh );
      MeshSuperentityStorageRebinder< Mesh, typename DimensionsTag::Decrement >::exec( mesh );
   }
};

template< typename Mesh >
struct MeshSuperentityStorageRebinder< Mesh, MeshDimensionsTag< 0 > >
{
   static void exec( Mesh& mesh )
   {
      MeshSuperentityStorageRebinderInner< Mesh, MeshDimensionsTag< 0 >, MeshDimensionsTag< Mesh::MeshTraitsType::meshDimensions > >::exec( mesh );
   }
};

} // namespace Meshes
} // namespace TNL
