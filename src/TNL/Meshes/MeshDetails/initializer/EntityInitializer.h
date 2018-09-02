/***************************************************************************
                          EntityInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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

#include <TNL/Meshes/MeshDetails/initializer/EntitySeed.h>
#include <TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>
#include <TNL/Meshes/MeshDetails/initializer/SuperentityStorageInitializer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class Initializer;

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag,
          // storage in the superentity
          bool SubentityStorage =
               MeshConfig::subentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >::EntityTopology(),
                                             SubdimensionTag::value ),
          bool SubentityOrientationStorage =
               MeshConfig::subentityOrientationStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >::EntityTopology(),
                                                        SubdimensionTag::value ) &&
               MeshTraits< MeshConfig >::template EntityTraits< SubdimensionTag::value >::orientationNeeded,
          // storage in the subentity
          bool SuperentityStorage =
               MeshConfig::superentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SubdimensionTag::value >::EntityTopology(),
                                               SuperdimensionTag::value ),
          // necessary to disambiguate the stop condition for specializations
          bool valid_dimension = ! std::is_same< SubdimensionTag, SuperdimensionTag >::value >
class EntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class EntityInitializer
   : public EntityInitializerLayer< MeshConfig,
                                    DimensionTag< EntityTopology::dimension >,
                                    DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            DimensionTag< EntityTopology::dimension >,
                                            DimensionTag< MeshTraits< MeshConfig >::meshDimension > >;

   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType   = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using EntityType       = typename EntityTraitsType::EntityType;

   using SeedType         = EntitySeed< MeshConfig, EntityTopology >;
   using InitializerType  = Initializer< MeshConfig >;

public:
   static String getType() { return "EntityInitializer"; };

   static void initEntity( EntityType& entity, const GlobalIndexType& entityIndex, const SeedType& entitySeed, InitializerType& initializer)
   {
      initializer.setEntityIndex( entity, entityIndex );
      // this is necessary if we want to use existing entities instead of intermediate seeds to create subentity seeds
      for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
         initializer.template setSubentityIndex< 0 >( entity, entityIndex, i, entitySeed.getCornerIds()[ i ] );
   }
};

template< typename MeshConfig >
class EntityInitializer< MeshConfig, Topologies::Vertex >
   : public EntityInitializerLayer< MeshConfig,
                                    DimensionTag< 0 >,
                                    DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
public:
   using VertexType      = typename MeshTraits< MeshConfig >::VertexType;
   using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using PointType       = typename MeshTraits< MeshConfig >::PointType;
   using InitializerType = Initializer< MeshConfig >;

   static String getType() { return "EntityInitializer"; };

   static void initEntity( VertexType& entity, const GlobalIndexType& entityIndex, const PointType& point, InitializerType& initializer)
   {
      initializer.setEntityIndex( entity, entityIndex );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              false,
                              true,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType            = Initializer< MeshConfig >;
   using MeshType                   = typename InitializerType::MeshType;

   using GlobalIndexType            = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType             = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType  = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;
   using SuperentityInitializerType = SuperentityStorageInitializer< MeshConfig, SubdimensionTag, SuperdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionTag::value >( superentity, superentityIndex, i, subentityIndex );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              true,
                              true,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType            = Initializer< MeshConfig >;
   using MeshType                   = typename InitializerType::MeshType;

   using GlobalIndexType            = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType             = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType  = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;
   using SuperentityInitializerType = SuperentityStorageInitializer< MeshConfig, SubdimensionTag, SuperdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         auto& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< SubdimensionTag >( superentity );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionTag::value >( superentity, superentityIndex, i, subentityIndex );
            superentityInitializer.addSuperentity( subentityIndex, superentityIndex );

            subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< SubdimensionTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              true,
                              false,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType           = Initializer< MeshConfig >;
   using MeshType                  = typename InitializerType::MeshType;

   using GlobalIndexType           = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType     = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology       = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;
      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         auto& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< SubdimensionTag >( superentity );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionTag::value >( superentity, superentityIndex, i, subentityIndex );

            subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< SubdimensionTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              false,
                              false,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType           = Initializer< MeshConfig >;
   using MeshType                  = typename InitializerType::MeshType;

   using GlobalIndexType           = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType     = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology       = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;
      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( superentityIndex );
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.getSubvertices( superentity, superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SubdimensionTag::value >( superentity, superentityIndex, i, subentityIndex );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              false,
                              false,
                              true,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType            = Initializer< MeshConfig >;
   using MeshType                   = typename InitializerType::MeshType;

   using GlobalIndexType            = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType             = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType  = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;
   using SuperentityInitializerType = SuperentityStorageInitializer< MeshConfig, SubdimensionTag, SuperdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;
      SuperentityInitializerType superentityInitializer;

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto& superentity = mesh.template getEntity< SuperdimensionTag::value >( superentityIndex );
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
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              false,
                              false,
                              false,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{};

template< typename MeshConfig,
          typename SubdimensionTag,
          bool SubentityStorage,
          bool SubentityOrientationStorage,
          bool SuperentityStorage >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SubdimensionTag,
                              SubentityStorage,
                              SubentityOrientationStorage,
                              SuperentityStorage,
                              false >
{
   using InitializerType = Initializer< MeshConfig >;
   using MeshType        = typename InitializerType::MeshType;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh ) {}
};

} // namespace Meshes
} // namespace TNL
