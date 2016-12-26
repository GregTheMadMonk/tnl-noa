/***************************************************************************
                          BoundaryTagsInitializer.h  -  description
                             -------------------
    begin                : Dec 26, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/StaticFor.h>
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>

namespace TNL {
namespace Meshes {

template< typename Mesh,
          typename CurrentDimension = DimensionTag< Mesh::getMeshDimension() > >
struct BoundaryTagsNeedInitialization
{
   using EntityTopology = typename MeshEntityTopology< typename Mesh::Config, CurrentDimension >::Topology;
   static constexpr bool value = Mesh::Config::boundaryTagsStorage( EntityTopology() ) ||
                                 BoundaryTagsNeedInitialization< Mesh, typename CurrentDimension::Decrement >::value;
};

template< typename Mesh >
struct BoundaryTagsNeedInitialization< Mesh, DimensionTag< 0 > >
{
   using EntityTopology = typename MeshEntityTopology< typename Mesh::Config, DimensionTag< 0 > >::Topology;
   static constexpr bool value = Mesh::Config::boundaryTagsStorage( EntityTopology() );
};


template< typename Mesh,
          bool AnyBoundaryTags = BoundaryTagsNeedInitialization< Mesh >::value >
class BoundaryTagsInitializer
{
   using GlobalIndexType = typename Mesh::Config::GlobalIndexType;
   using LocalIndexType  = typename Mesh::Config::LocalIndexType;
   using FaceTraits      = typename MeshTraits< typename Mesh::Config >::template EntityTraits< Mesh::getMeshDimension() - 1 >;
   using FaceType        = typename FaceTraits::EntityType;

public:
   template< typename MeshInitializer >
   static bool exec( MeshInitializer& initializer, Mesh& mesh )
   {
      StaticFor< int, 0, Mesh::getMeshDimension() + 1, ResetBoundaryTags >::exec( initializer );

      const GlobalIndexType facesCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() - 1 >();
      for( GlobalIndexType faceIndex = 0; faceIndex < facesCount; faceIndex++ ) {
         const auto& face = mesh.template getEntity< Mesh::getMeshDimension() - 1 >( faceIndex );
         if( face.template getSuperentitiesCount< Mesh::getMeshDimension() >() == 1 ) {
            // initialize the face
            initializer.template setIsBoundaryEntity< Mesh::getMeshDimension() - 1 >( faceIndex, true );
            // initialize the cell superentity
            const GlobalIndexType cellIndex = face.template getSuperentityIndex< Mesh::getMeshDimension() >( 0 );
            initializer.template setIsBoundaryEntity< Mesh::getMeshDimension() >( cellIndex, true );
            // initialize all subentities
            StaticFor< int, 0, Mesh::getMeshDimension() - 1, InitializeSubentities >::exec( initializer, faceIndex, face );
         }
      }

      // hack due to StaticFor operating only with void return type
      bool result = true;

      StaticFor< int, 0, Mesh::getMeshDimension() + 1, UpdateBoundaryIndices >::exec( initializer, result );

      return result;
   }

private:
   template< int Dimension >
   struct ResetBoundaryTags
   {
      template< typename MeshInitializer >
      static void exec( MeshInitializer& initializer )
      {
         initializer.template resetBoundaryTags< Dimension >();
      }
   };

   template< int Subdimension >
   class InitializeSubentities
   {
      using SubentityTopology = typename MeshEntityTopology< typename Mesh::Config, DimensionTag< Subdimension > >::Topology;
      static constexpr bool enabled = Mesh::Config::boundaryTagsStorage( SubentityTopology() );

      // _T is necessary to force *partial* specialization, since explicit specializations
      // at class scope are forbidden
      template< bool enabled = true, typename _T = void >
      struct Worker
      {
         template< typename MeshInitializer >
         static void exec( MeshInitializer& initializer, const GlobalIndexType& faceIndex, const FaceType& face )
         {
            auto subentitiesCount = face.template getSubentitiesCount< Subdimension >();
            for( decltype(subentitiesCount) i = 0; i < subentitiesCount; i++ ) {
               const GlobalIndexType subentityIndex = face.template getSubentityIndex< Subdimension >( i );
               initializer.template setIsBoundaryEntity< Subdimension >( subentityIndex, true );
            }
         }
      };

      template< typename _T >
      struct Worker< false, _T >
      {
         template< typename MeshInitializer >
         static void exec( MeshInitializer& initializer, const GlobalIndexType& faceIndex, const FaceType& face )
         {}
      };

      public:
         template< typename MeshInitializer >
         static void
         exec( MeshInitializer& initializer, const GlobalIndexType& faceIndex, const FaceType& face )
         {
            Worker< enabled >::exec( initializer, faceIndex, face );
         }
   };

   template< int Dimension >
   struct UpdateBoundaryIndices
   {
      template< typename MeshInitializer >
      static void exec( MeshInitializer& initializer, bool& result )
      {
         result &= initializer.template updateBoundaryIndices< Dimension >();
      }
   };
};

template< typename Mesh >
struct BoundaryTagsInitializer< Mesh, false >
{
   template< typename MeshInitializer >
   static bool exec( MeshInitializer& initializer, Mesh& mesh )
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
