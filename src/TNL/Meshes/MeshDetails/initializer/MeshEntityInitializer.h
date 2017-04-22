/***************************************************************************
                          MeshEntityInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
                                                              typename MeshSubentityTraits< MeshConfig, EntityTopology, DimensionTag::value >::SubentityTopology,
                                                              EntityTopology::dimensions >::storageEnabled >
class MeshEntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializer
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           MeshDimensionTag< EntityTopology::dimensions - 1 > >
{
   typedef MeshDimensionTag< EntityTopology::dimensions >                                 DimensionTag;
   private:

      typedef MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           MeshDimensionTag< EntityTopology::dimensions - 1 > > BaseType;
 
   typedef
      MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     MeshDimensionTag< EntityTopology::dimensions - 1 > >   SubentityBaseType;
   typedef
      MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename
                                          MeshTraits< MeshConfig >::DimensionTag > SuperentityBaseType;

   static const int Dimension = DimensionTag::value;
   typedef MeshTraits< MeshConfig >                                                 MeshTraitsType;
   typedef typename MeshTraitsType::GlobalIndexType                                 GlobalIndexType;
   typedef typename MeshTraitsType::LocalIndexType                                  LocalIndexType;
   typedef typename MeshTraitsType::template EntityTraits< Dimension >             EntityTraitsType;
 
   typedef typename EntityTraitsType::EntityType                                    EntityType;
   typedef typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >   SubvertexTraits;
 
   typedef MeshInitializer< MeshConfig >                                            InitializerType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                             SeedType;

   template< typename > class SubentitiesCreator;

   public:

   //using SuperentityBaseType::setNumberOfSuperentities;

   static String getType() { return "MeshEntityInitializer"; };

   MeshEntityInitializer() : entity(0), entityIndex( -1 ) {}

   static void initEntity( EntityType &entity, GlobalIndexType entityIndex, const SeedType &entitySeed, InitializerType &initializer)
   {
      entity = EntityType( entitySeed );
      BaseType::initSubentities( entity, entityIndex, entitySeed, initializer );
   }
 
   template< typename SuperentityDimensionTag >
   typename MeshSuperentityTraits< MeshConfig, EntityTopology, SuperentityDimensionTag::value >::SharedContainerType& getSuperentityContainer( SuperentityDimensionTag )
   {
      return this->entity->template getSuperentitiesIndices< SuperentityDimensionTag::value >();
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
      typedef typename MeshTraits< MeshConfig >::VertexType VertexType;
      typedef typename MeshTraits< MeshConfig >::PointType  PointType;
      typedef MeshInitializer< MeshConfig >                 InitializerType;

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
                                     DimensionTag,
                                     true,
                                     false,
                                     true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{
   typedef MeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionTag::Decrement >                BaseType;

   static const int Dimension = DimensionTag::value;
   typedef MeshTraits< MeshConfig >                                                          MeshTraitsType;
   typedef typename MeshTraitsType::template SubentityTraits< EntityTopology, Dimension >   SubentityTraitsType;
   typedef typename SubentityTraitsType::SubentityContainerType                              SubentityContainerType;
   typedef typename SubentityTraitsType::AccessArrayType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                         GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                                                     InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                               EntityInitializerType;
   typedef MeshDimensionTag< EntityTopology::dimensions >                                    EntityDimensionTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                                          EntityType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                                      SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >            SubentitySeedsCreatorType;
   typedef typename SubentityTraitsType::IdArrayType                                         IdArrayType;
   typedef typename MeshTraitsType::LocalIndexType                                           LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
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
                                     DimensionTag,
                                     true,
                                     true,
                                     true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{
   typedef MeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionTag::Decrement >                   BaseType;

   typedef MeshSubentityTraits< MeshConfig, EntityTopology, DimensionTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::AccessArrayType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                                                          InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                                         EntityInitializerType;
   typedef MeshDimensionTag< EntityTopology::dimensions >                                                EntityDimensionTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                                                SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                      SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag::value >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityIdsArray[ i ] = subentityIndex;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         //cout << "    Subentity orientation = " << subentityOrientationsArray[ i ].getSubvertexPermutation() << std::endl;
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
                                     DimensionTag,
                                     true,
                                     true,
                                     false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{
   typedef MeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionTag::Decrement >                   BaseType;

   typedef MeshSubentityTraits< MeshConfig, EntityTopology, DimensionTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                                                          InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                                         EntityInitializerType;
   typedef MeshDimensionTag< EntityTopology::dimensions >                                                EntityDimensionTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                                                SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                      SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionTag >( subentitySeeds[ i ] ).createOrientation( subentitySeeds[ i ] );
      }
      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionTag,
                                     true,
                                     false,
                                     false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{
   typedef MeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionTag::Decrement >                   BaseType;

   typedef typename MeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionTag::value >::SubentityContainerType          SubentityContainerType;
   typedef typename MeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionTag::value >::SharedContainerType             SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                                                     InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                                    EntityInitializerType;
   typedef MeshEntity< MeshConfig, EntityTopology >                                               EntityType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                 SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
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
                                     DimensionTag,
                                     false,
                                     false,
                                     true >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{
   typedef MeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionTag::Decrement >                BaseType;

   typedef typename MeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionTag::value >::SubentityContainerType        SubentityContainerType;
   typedef typename MeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionTag::value >::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::DataType                                            GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                                                   InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                                  EntityInitializerType;
   typedef MeshDimensionTag< EntityTopology::dimensions >                                      EntityDimensionTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                                           EntityType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                 SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
         auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );
         IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >( entity );
         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
         {
            GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.
               template getSuperentityInitializer< DimensionTag >().
                  addSuperentity( EntityDimensionTag(), subentityIndex, entityIndex );
         }
         BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
      }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionTag,
                                     false,
                                     false,
                                     false >
   : public MeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionTag::Decrement >
{};

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     MeshDimensionTag< 0 >,
                                     true,
                                     false,
                                     true >
{
   typedef MeshDimensionTag< 0 >                                  DimensionTag;
   typedef MeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionTag::value >                 SubentitiesTraits;

   typedef typename SubentitiesTraits::AccessArrayType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;

   typedef MeshInitializer< MeshConfig >                           InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >          EntityInitializerType;
   typedef MeshDimensionTag< EntityTopology::dimensions >              EntityDimensionTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
      typedef MeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                 SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag::value >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         //cout << "   Initiating subentities with " << DimensionTag::value << " dimensions ... " << std::endl;
		   const IdArrayType &subentityIdsArray = InitializerType::template subentityIdsArray< DimensionTag >(entity);
		   for( LocalIndexType i = 0; i < subentityIdsArray.getSize(); i++ )
			   meshInitializer.template getSuperentityInitializer< DimensionTag >().addSuperentity( EntityDimensionTag(), subentityIdsArray[ i ], entityIndex);
	}

};

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     MeshDimensionTag< 0 >,
                                     true,
                                     false,
                                     false >
{
   typedef MeshInitializer< MeshConfig >         InitializerType;
   typedef MeshEntityInitializer< MeshConfig,
                                     EntityTopology >   EntityInitializerType;
   typedef MeshDimensionTag< 0 >                   DimensionTag;
   typedef MeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
      typedef MeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef MeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag >                 SubentitySeedsCreatorType;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionTag >::IdArrayType IdArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};
 
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     MeshDimensionTag< 0 >,
                                     false,
                                     true,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef MeshInitializer< MeshConfig >                  InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology > EntityInitializerType;
   typedef MeshDimensionTag< 0 >                   DimensionTag;
   typedef MeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&, InitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class MeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     MeshDimensionTag< 0 >,
                                     false,
                                     false,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef MeshInitializer< MeshConfig >                  InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology > EntityInitializerType;
   typedef MeshDimensionTag< 0 >                   DimensionTag;
   typedef MeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef MeshEntity< MeshConfig, EntityTopology >                                                    EntityType;

   protected:
   void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&,
                         InitializerType& ) {}
};

} // namespace Meshes
} // namespace TNL
