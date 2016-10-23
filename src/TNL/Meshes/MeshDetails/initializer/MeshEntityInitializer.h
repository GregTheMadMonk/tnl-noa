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
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag,
          bool SubentityStorage = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionTag::value >::storageEnabled,
          bool SubentityOrientationStorage = MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag::value >::orientationEnabled,
          bool SuperentityStorage = MeshSuperentityTraits< MeshConfig,
                                                           typename MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::SubentityTopology,
                                                           EntityTopology::dimensions >::storageEnabled >
class MeshEntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializer
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        MeshDimensionsTag< EntityTopology::dimensions - 1 > >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                MeshDimensionsTag< EntityTopology::dimensions - 1 > >;
   using SubentityBaseType = MeshEntityInitializerLayer< MeshConfig,
                                                         EntityTopology,
                                                         MeshDimensionsTag< EntityTopology::dimensions - 1 > >;
   using SuperentityBaseType = MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                                       EntityTopology,
                                                                       typename MeshTraits< MeshConfig >::DimensionsTag >;

   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType   = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< Dimensions >;

   using EntityType       = typename EntityTraitsType::EntityType;
   using SubvertexTraits  = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;

   using InitializerType  = MeshInitializer< MeshConfig >;
   using SeedType         = MeshEntitySeed< MeshConfig, EntityTopology >;

public:
   static String getType() { return "MeshEntityInitializer"; };

   MeshEntityInitializer() : entity(0), entityIndex( -1 ) {}

   static void initEntity( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer)
   {
      for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
         initializer.template setSubentityIndex< 0 >( entity, i, entitySeed.getCornerIds()[ i ] );
      BaseType::initSubentities( entity, entityIndex, entitySeed, initializer );
   }

   static void setEntityVertex( EntityType& entity,
                                LocalIndexType localIndex,
                                GlobalIndexType globalIndex )
   {
      entity.setVertexIndex( localIndex, globalIndex );
   }

private:
   EntityType *entity;
   GlobalIndexType entityIndex;
};

template< typename MeshConfig >
class MeshEntityInitializer< MeshConfig, MeshVertexTopology >
{
public:
   using VertexType      = typename MeshTraits< MeshConfig >::VertexType;
   using PointType       = typename MeshTraits< MeshConfig >::PointType;
   using InitializerType = MeshInitializer< MeshConfig >;

   static String getType() { return "MeshEntityInitializer"; };

   static void setVertexPoint( VertexType& vertex,
                               const PointType& point,
                               InitializerType& initializer )
   {
      initializer.setVertexPoint( vertex, point );
   }
};


/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    FALSE                    TRUE
 */
template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  true,
                                  false,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >;

   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType            = MeshTraits< MeshConfig >;
   using SubentityTraitsType       = typename MeshTraitsType::template SubentityTraits< EntityTopology, Dimensions >;
   using SubentityContainerType    = typename SubentityTraitsType::SubentityContainerType;
   using SharedContainerType       = typename SubentityTraitsType::AccessArrayType;
   using GlobalIndexType           = typename SharedContainerType::ElementType;

   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityDimensionsTag       = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename SubentityTraitsType::IdArrayType;
   using LocalIndexType            = typename MeshTraitsType::LocalIndexType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         //std::cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         meshInitializer.
            template getSuperentityInitializer< DimensionTag >().
               addSuperentity( EntityDimensionTag(), subentityIdsArray[ i ], entityIndex );
      }

      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    TRUE                    TRUE
 */
template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  true,
                                  true,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >;

   using SubentitiesTraits         = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >;
   using SubentityContainerType    = typename SubentitiesTraits::SubentityContainerType;
   using SharedContainerType       = typename SubentitiesTraits::AccessArrayType;
   using GlobalIndexType           = typename SharedContainerType::ElementType;

   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityDimensionsTag       = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::IdArrayType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using OrientationArrayType      = typename SubentitiesTraits::OrientationArrayType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityIdsArray[ i ] = subentityIndex;
         //std::cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         //std::cout << "    Subentity orientation = " << subentityOrientationsArray[ i ].getSubvertexPermutation() << std::endl;
         meshInitializer.
            template getSuperentityInitializer< DimensionTag >().
               addSuperentity( EntityDimensionTag(), subentityIdsArray[ i ], entityIndex );
      }

      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUBENTITY ORIENTATION    SUPERENTITY STORAGE
 *      TRUE                    TRUE                    FALSE
 */
template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  true,
                                  true,
                                  false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >;

   using SubentitiesTraits         = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >;
   using SubentityContainerType    = typename SubentitiesTraits::SubentityContainerType;
   using SharedContainerType       = typename SubentitiesTraits::SharedContainerType;
   using GlobalIndexType           = typename SharedContainerType::ElementType;

   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityDimensionsTag       = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using OrientationArrayType      = typename SubentitiesTraits::OrientationArrayType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType& subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         //std::cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentitySeeds[ i ] ).createOrientation( subentitySeeds[ i ] );
      }

      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  true,
                                  false,
                                  false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >;
   using SubentityContainerType = typename MeshSubentityTraits< MeshConfig,
                                                                EntityTopology,
                                                                DimensionsTag::value >::SubentityContainerType;
   using SharedContainerType = typename MeshSubentityTraits< MeshConfig,
                                                             EntityTopology,
                                                             DimensionsTag::value >::SharedContainerType;

   using GlobalIndexType           = typename SharedContainerType::ElementType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

		IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
		for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
			subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );

		BaseType::initSubentities(entity, entityIndex, entitySeed, meshInitializer);
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  false,
                                  false,
                                  true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{
   using BaseType = MeshEntityInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >;
   using SubentityContainerType = typename MeshSubentityTraits< MeshConfig,
                                                                EntityTopology,
                                                                DimensionsTag::value >::SubentityContainerType;
   using SharedContainerType = typename MeshSubentityTraits< MeshConfig,
                                                             EntityTopology,
                                                             DimensionsTag::value >::SharedContainerType;

   using GlobalIndexType           = typename SharedContainerType::DataType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityDimensionsTag       = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );
      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         meshInitializer.
            template getSuperentityInitializer< DimensionsTag >().
               addSuperentity( EntityDimensionsTag(), subentityIndex, entityIndex );
      }

      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  false,
                                  false,
                                  false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                        EntityTopology,
                                        typename DimensionsTag::Decrement >
{};

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< 0 >,
                                  true,
                                  false,
                                  true >
{
   using DimensionsTag = MeshDimensionsTag< 0 >;
   using SubentitiesTraits = MeshSubentityTraits< MeshConfig,
                                                  EntityTopology,
                                                  DimensionsTag::value >;

   using SharedContainerType       = typename SubentitiesTraits::AccessArrayType;
   using GlobalIndexType           = typename SharedContainerType::ElementType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using InitializerType           = MeshInitializer< MeshConfig >;
   using EntityInitializerType     = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntityDimensionsTag       = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SubentitySeedsCreatorType = MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >;
   using IdArrayType               = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::IdArrayType;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //std::cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      const IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >(entity);
      for( LocalIndexType i = 0; i < subentityIdsArray.getSize(); i++ )
         meshInitializer.template getSuperentityInitializer< DimensionsTag >().addSuperentity( EntityDimensionsTag(), subentityIdsArray[ i ], entityIndex);
	}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< 0 >,
                                  true,
                                  false,
                                  false >
{
   using InitializerType           = MeshInitializer< MeshConfig >;
   using DimensionsTag             = MeshDimensionsTag< 0 >;
   using SubentitiesTraits         = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >;
   using SharedContainerType       = typename SubentitiesTraits::SharedContainerType;
   using GlobalIndexType           = typename SharedContainerType::ElementType;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                  = MeshEntitySeed< MeshConfig, EntityTopology >;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer ) {}
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< 0 >,
                                  false,
                                  true,
                                  SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   using InitializerType       = MeshInitializer< MeshConfig >;
   using EntityInitializerType = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using DimensionsTag         = MeshDimensionsTag< 0 >;
   using SubentitiesTraits     = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >;
   using SharedContainerType   = typename SubentitiesTraits::SharedContainerType;
   using GlobalIndexType       = typename SharedContainerType::ElementType;
   using EntityType            = MeshEntity< MeshConfig, EntityTopology >;

protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&, InitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class MeshEntityInitializerLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< 0 >,
                                  false,
                                  false,
                                  SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   using InitializerType       = MeshInitializer< MeshConfig >;
   using EntityInitializerType = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using DimensionsTag         = MeshDimensionsTag< 0 >;
   using SubentitiesTraits     = MeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >;
   using SharedContainerType   = typename SubentitiesTraits::SharedContainerType;
   using GlobalIndexType       = typename SharedContainerType::ElementType;
   using EntityType            = MeshEntity< MeshConfig, EntityTopology >;

protected:
   void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&,
                         InitializerType& ) {}
};

} // namespace Meshes
} // namespace TNL
