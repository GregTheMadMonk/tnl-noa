/***************************************************************************
                          Initializer.h  -  description
                             -------------------
    begin                : Dec 26, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Algorithms/TemplateStaticFor.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>

#include "Traits.h"

namespace TNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig, typename Device, typename Mesh >
class Initializer
{
   using DeviceType      = Device;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

protected:
   // _T is necessary to force *partial* specialization, since explicit specializations
   // at class scope are forbidden
   template< typename CurrentDimension = DimensionTag< MeshConfig::meshDimension >, typename _T = void >
   struct EntityTagsNeedInitialization
   {
      using EntityTopology = typename MeshEntityTraits< MeshConfig, DeviceType, CurrentDimension::value >::EntityTopology;
      static constexpr bool value = MeshConfig::entityTagsStorage( EntityTopology() ) ||
                                    EntityTagsNeedInitialization< typename CurrentDimension::Decrement >::value;
   };

   template< typename _T >
   struct EntityTagsNeedInitialization< DimensionTag< 0 >, _T >
   {
      using EntityTopology = typename MeshEntityTraits< MeshConfig, DeviceType, 0 >::EntityTopology;
      static constexpr bool value = MeshConfig::entityTagsStorage( EntityTopology() );
   };

   template< int Dimension >
   struct SetEntitiesCount
   {
      static void exec( Mesh& mesh )
      {
         mesh.template entityTagsSetEntitiesCount< Dimension >( mesh.template getEntitiesCount< Dimension >() );
      }
   };

   template< int Dimension >
   struct ResetEntityTags
   {
      static void exec( Mesh& mesh )
      {
         mesh.template resetEntityTags< Dimension >();
      }
   };

   template< int Subdimension >
   class InitializeSubentities
   {
      using SubentityTopology = typename MeshEntityTraits< MeshConfig, DeviceType, Subdimension >::EntityTopology;
      static constexpr bool enabled = MeshConfig::entityTagsStorage( SubentityTopology() );

      // _T is necessary to force *partial* specialization, since explicit specializations
      // at class scope are forbidden
      template< bool enabled = true, typename _T = void >
      struct Worker
      {
         __cuda_callable__
         static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face )
         {
            const LocalIndexType subentitiesCount = face.template getSubentitiesCount< Subdimension >();
            for( LocalIndexType i = 0; i < subentitiesCount; i++ ) {
               const GlobalIndexType subentityIndex = face.template getSubentityIndex< Subdimension >( i );
               mesh.template addEntityTag< Subdimension >( subentityIndex, EntityTags::BoundaryEntity );
            }
         }
      };

      template< typename _T >
      struct Worker< false, _T >
      {
         __cuda_callable__
         static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face ) {}
      };

   public:
      __cuda_callable__
      static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face )
      {
         Worker< enabled >::exec( mesh, faceIndex, face );
      }
   };

   template< int Dimension >
   struct UpdateEntityTagsLayer
   {
      static void exec( Mesh& mesh )
      {
         mesh.template updateEntityTagsLayer< Dimension >();
      }
   };

// nvcc does not allow __cuda_callable__ lambdas inside private or protected sections
#ifdef __NVCC__
public:
#endif
   // _T is necessary to force *partial* specialization, since explicit specializations
   // at class scope are forbidden
   template< bool AnyEntityTags = EntityTagsNeedInitialization<>::value, typename _T = void >
   class Worker
   {
   public:
      static void exec( Mesh& mesh )
      {
         Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() + 1, SetEntitiesCount >::execHost( mesh );
         Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() + 1, ResetEntityTags >::execHost( mesh );

         auto kernel = [] __cuda_callable__
            ( GlobalIndexType faceIndex,
              Mesh* mesh )
         {
            const auto& face = mesh->template getEntity< Mesh::getMeshDimension() - 1 >( faceIndex );
            if( face.template getSuperentitiesCount< Mesh::getMeshDimension() >() == 1 ) {
               // initialize the face
               mesh->template addEntityTag< Mesh::getMeshDimension() - 1 >( faceIndex, EntityTags::BoundaryEntity );
               // initialize the cell superentity
               const GlobalIndexType cellIndex = face.template getSuperentityIndex< Mesh::getMeshDimension() >( 0 );
               mesh->template addEntityTag< Mesh::getMeshDimension() >( cellIndex, EntityTags::BoundaryEntity );
               // initialize all subentities
               Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() - 1, InitializeSubentities >::exec( *mesh, faceIndex, face );
            }
         };

         const GlobalIndexType facesCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() - 1 >();
         Pointers::DevicePointer< Mesh > meshPointer( mesh );
         Algorithms::ParallelFor< DeviceType >::exec( (GlobalIndexType) 0, facesCount,
                                                      kernel,
                                                      &meshPointer.template modifyData< DeviceType >() );

         Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() + 1, UpdateEntityTagsLayer >::execHost( mesh );
      }
   };

   template< typename _T >
   struct Worker< false, _T >
   {
      static void exec( Mesh& mesh ) {}
   };

public:
   void initLayer()
   {
      Worker<>::exec( *static_cast<Mesh*>(this) );
   }
};

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
