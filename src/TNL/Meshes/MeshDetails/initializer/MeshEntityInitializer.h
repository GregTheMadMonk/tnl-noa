/***************************************************************************
                          MeshEntityInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/StaticFor.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshSuperentityStorageInitializer.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshSubentitySeedCreator.h>

#include "MeshEntitySeed.h"

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class Mesh;

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag,
          // storage in the superentity
          bool SubentityStorage =
               MeshConfig::subentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >::EntityTopology(),
                                             SubdimensionsTag::value ), 
          bool SubentityOrientationStorage =
             // FIXME
             false,
//             MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::orientationEnabled &&
//             MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::orientationNeeded,
          // storage in the subentity
          bool SuperentityStorage =
               MeshConfig::superentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SubdimensionsTag::value >::EntityTopology(),
                                               SuperdimensionsTag::value ),
          // necessary to disambiguate the stop condition for specializations
          bool valid_dimensions = ! std::is_same< SubdimensionsTag, SuperdimensionsTag >::value >
class MeshEntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializer
   : public MeshEntityInitializerLayer< MeshConfig,
                                        MeshDimensionsTag< EntityTopology::dimensions >,
                                        MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                MeshDimensionsTag< EntityTopology::dimensions >,
                                                MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >;

   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType   = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityTopology::dimensions >;
   using EntityType       = typename EntityTraitsType::EntityType;

   using SeedType         = MeshEntitySeed< MeshConfig, EntityTopology >;
   using InitializerType  = MeshInitializer< MeshConfig >;

public:
   static String getType() { return "MeshEntityInitializer"; };

   static void initEntity( EntityType& entity, const GlobalIndexType& entityIndex, const SeedType& entitySeed, InitializerType& initializer)
   {
      initializer.setEntityId( entity, entityIndex );
      // this is necessary if we want to use existing entities instead of intermediate seeds to create subentity seeds
      for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
         initializer.template setSubentityIndex< 0 >( entity, entityIndex, i, entitySeed.getCornerIds()[ i ] );
   }
};

template< typename MeshConfig >
class MeshEntityInitializer< MeshConfig, MeshVertexTopology >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        MeshDimensionsTag< 0 >,
                                        MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{
public:
   using VertexType      = typename MeshTraits< MeshConfig >::VertexType;
   using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using PointType       = typename MeshTraits< MeshConfig >::PointType;
   using InitializerType = MeshInitializer< MeshConfig >;

   static String getType() { return "MeshEntityInitializer"; };

   static void initEntity( VertexType& entity, const GlobalIndexType& entityIndex, const PointType& point, InitializerType& initializer)
   {
      initializer.setEntityId( entity, entityIndex );
      initializer.setVertexPoint( entity, point );
   }
};


/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    FALSE                    TRUE
 */
template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  true,
                                  false,
                                  true,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                SubdimensionsTag,
                                                typename SuperdimensionsTag::Decrement >;
   using InitializerType            = MeshInitializer< MeshConfig >;
   using MeshType                   = Mesh< MeshConfig >;

   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using GlobalIndexType            = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType             = typename SuperentityTraitsType::LocalIndexType;
   using SubentitySeedsCreatorType  = MeshSubentitySeedsCreator< MeshConfig, SuperdimensionsTag, SubdimensionsTag >;
   using SuperentityInitializerType = MeshSuperentityStorageInitializer< MeshConfig, SubdimensionsTag, SuperdimensionsTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with " << SuperdimensionsTag::value << " dimensions for subentities with " << SubdimensionsTag::value << " dimensions ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getNumberOfEntities< SuperdimensionsTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionsTag::value >( superentity, superentityIndex, i, subentityIndex );
            superentityInitializer.addSuperentity( subentityIndex, superentityIndex );
         }
      }

      superentityInitializer.initSuperentities( meshInitializer );

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    TRUE                    TRUE
 */
template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  true,
                                  true,
                                  true,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                SubdimensionsTag,
                                                typename SuperdimensionsTag::Decrement >;
   using InitializerType            = MeshInitializer< MeshConfig >;
   using MeshType                   = Mesh< MeshConfig >;

   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using GlobalIndexType            = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType             = typename SuperentityTraitsType::LocalIndexType;
   using SubentitySeedsCreatorType  = MeshSubentitySeedsCreator< MeshConfig, SuperdimensionsTag, SubdimensionsTag >;
   using SuperentityInitializerType = MeshSuperentityStorageInitializer< MeshConfig, SubdimensionsTag, SuperdimensionsTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with " << SuperdimensionsTag::value << " dimensions for subentities with " << SubdimensionsTag::value << " dimensions ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getNumberOfEntities< SuperdimensionsTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         auto& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< SuperdimensionsTag >( superentity );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionsTag::value >( superentity, superentityIndex, i, subentityIndex );
            superentityInitializer.addSuperentity( subentityIndex, superentityIndex );

            subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< SuperdimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         }
      }

      superentityInitializer.initSuperentities( meshInitializer );

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    TRUE                    FALSE
 */
template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  true,
                                  true,
                                  false,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                SubdimensionsTag,
                                                typename SuperdimensionsTag::Decrement >;
   using InitializerType           = MeshInitializer< MeshConfig >;
   using MeshType                  = Mesh< MeshConfig >;

   using SuperentityTraitsType     = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >;
   using SuperentityTopology       = typename SuperentityTraitsType::EntityTopology;
   using GlobalIndexType           = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType            = typename SuperentityTraitsType::LocalIndexType;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, SuperdimensionsTag, SubdimensionsTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with " << SuperdimensionsTag::value << " dimensions for subentities with " << SubdimensionsTag::value << " dimensions ... " << std::endl;
      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getNumberOfEntities< SuperdimensionsTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         auto& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< SuperdimensionsTag >( superentity );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionsTag::value >( superentity, superentityIndex, i, subentityIndex );

            subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< SuperdimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    FALSE                   FALSE
 */
template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  true,
                                  false,
                                  false,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                SubdimensionsTag,
                                                typename SuperdimensionsTag::Decrement >;
   using InitializerType           = MeshInitializer< MeshConfig >;
   using MeshType                  = Mesh< MeshConfig >;

   using SuperentityTraitsType     = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >;
   using SuperentityTopology       = typename SuperentityTraitsType::EntityTopology;
   using GlobalIndexType           = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType            = typename SuperentityTraitsType::LocalIndexType;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, SuperdimensionsTag, SubdimensionsTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with " << SuperdimensionsTag::value << " dimensions for subentities with " << SubdimensionsTag::value << " dimensions ... " << std::endl;
      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getNumberOfEntities< SuperdimensionsTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionsTag::value >( superentity, superentityIndex, i, subentityIndex );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      FALSE                   FALSE                   TRUE
 */
template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  false,
                                  false,
                                  true,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                SubdimensionsTag,
                                                typename SuperdimensionsTag::Decrement >;
   using InitializerType            = MeshInitializer< MeshConfig >;
   using MeshType                   = Mesh< MeshConfig >;

   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionsTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using GlobalIndexType            = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType             = typename SuperentityTraitsType::LocalIndexType;
   using SubentitySeedsCreatorType  = MeshSubentitySeedsCreator< MeshConfig, SuperdimensionsTag, SubdimensionsTag >;
   using SuperentityInitializerType = MeshSuperentityStorageInitializer< MeshConfig, SubdimensionsTag, SuperdimensionsTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with " << SuperdimensionsTag::value << " dimensions for subentities with " << SubdimensionsTag::value << " dimensions ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getNumberOfEntities< SuperdimensionsTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionsTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            superentityInitializer.addSuperentity( subentityIndex, superentityIndex );
         }
      }

      superentityInitializer.initSuperentities( meshInitializer );

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

template< typename MeshConfig,
          typename SubdimensionsTag,
          typename SuperdimensionsTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SuperdimensionsTag,
                                  false,
                                  false,
                                  false,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        SubdimensionsTag,
                                        typename SuperdimensionsTag::Decrement >
{};

template< typename MeshConfig,
          typename SubdimensionsTag,
          bool SubentityStorage,
          bool SubentityOrientationStorage,
          bool SuperentityStorage >
class MeshEntityInitializerLayer< MeshConfig,
                                  SubdimensionsTag,
                                  SubdimensionsTag,
                                  SubentityStorage,
                                  SubentityOrientationStorage,
                                  SuperentityStorage,
                                  false >
{
   using InitializerType = MeshInitializer< MeshConfig >;
   using MeshType = Mesh< MeshConfig >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh ) {}
};

} // namespace Meshes
} // namespace TNL
