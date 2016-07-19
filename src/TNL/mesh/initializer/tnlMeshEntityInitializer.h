/***************************************************************************
                          tnlMeshEntityInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlStaticFor.h>
#include <TNL/mesh/initializer/tnlMeshSuperentityStorageInitializer.h>
#include <TNL/mesh/initializer/tnlMeshSubentitySeedCreator.h>

#include "tnlMeshEntitySeed.h"

namespace TNL {

template< typename MeshConfig >
class tnlMeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SubentityStorage = tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled,
          bool SubentityOrientationStorage = tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::orientationEnabled,
          bool SuperentityStorage = tnlMeshSuperentityTraits< MeshConfig,
                                                              typename tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::SubentityTopology,
                                                              EntityTopology::dimensions >::storageEnabled >
class tnlMeshEntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntityInitializer
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           tnlDimensionsTag< EntityTopology::dimensions - 1 > >
{
   typedef tnlDimensionsTag< EntityTopology::dimensions >                                 DimensionsTag;
   private:

      typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           tnlDimensionsTag< EntityTopology::dimensions - 1 > > BaseType;
 
   typedef
      tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     tnlDimensionsTag< EntityTopology::dimensions - 1 > >   SubentityBaseType;
   typedef
      tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename
                                          tnlMeshTraits< MeshConfig >::DimensionsTag > SuperentityBaseType;

   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshTraits< MeshConfig >                                          MeshTraits;
   typedef typename MeshTraits::GlobalIndexType                                 GlobalIndexType;
   typedef typename MeshTraits::LocalIndexType                                  LocalIndexType;
   typedef typename MeshTraits::template EntityTraits< Dimensions >             EntityTraits;
 
   typedef typename EntityTraits::EntityType                                    EntityType;
   typedef typename MeshTraits::template SubentityTraits< EntityTopology, 0 >   SubvertexTraits;
 

   typedef tnlMeshInitializer< MeshConfig >                                                   InitializerType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                            SeedType;

   template< typename > class SubentitiesCreator;

   public:

   //using SuperentityBaseType::setNumberOfSuperentities;

   static tnlString getType() {};

   tnlMeshEntityInitializer() : entity(0), entityIndex( -1 ) {}

   static void initEntity( EntityType &entity, GlobalIndexType entityIndex, const SeedType &entitySeed, InitializerType &initializer)
   {
      entity = EntityType( entitySeed );
      BaseType::initSubentities( entity, entityIndex, entitySeed, initializer );
   }
 
   template< typename SuperentityDimensionTag >
   typename tnlMeshSuperentityTraits< MeshConfig, EntityTopology, SuperentityDimensionTag::value >::SharedContainerType& getSuperentityContainer( SuperentityDimensionTag )
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
class tnlMeshEntityInitializer< MeshConfig, tnlMeshVertexTopology >
{
   public:
      typedef typename tnlMeshTraits< MeshConfig >::VertexType VertexType;
      typedef typename tnlMeshTraits< MeshConfig >::PointType  PointType;
      typedef tnlMeshInitializer< MeshConfig >                 InitializerType;

      static tnlString getType() {};
 
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
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     true,
                                     false,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >                   BaseType;

   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshTraits< MeshConfig >                                                          MeshTraits;
   typedef typename MeshTraits:: template SubentityTraits< EntityTopology, Dimensions >         SubentityTraits;
   typedef typename SubentityTraits::SubentityContainerType                                     SubentityContainerType;
   typedef typename SubentityTraits::AccessArrayType                                        SharedContainerType;
   typedef typename SharedContainerType::ElementType                                            GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                               EntityInitializerType;
   typedef tnlDimensionsTag< EntityTopology::dimensions >                                       EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                          EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                      SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >            SubentitySeedsCreatorType;
   typedef typename SubentityTraits::IdArrayType                                                IdArrayType;
   typedef typename MeshTraits::LocalIndexType                                                  LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         meshInitializer.
            template getSuperentityInitializer< DimensionsTag >().
               addSuperentity( EntityDimensionsTag(), subentityIdsArray[ i ], entityIndex );
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
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     true,
                                     true,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::AccessArrayType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTopology::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityIdsArray[ i ] = subentityIndex;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         //cout << "    Subentity orientation = " << subentityOrientationsArray[ i ].getSubvertexPermutation() << std::endl;
         meshInitializer.
            template getSuperentityInitializer< DimensionsTag >().
               addSuperentity( EntityDimensionsTag(), subentityIdsArray[ i ], entityIndex );
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
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     true,
                                     true,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTopology::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << std::endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentitySeeds[ i ] ).createOrientation( subentitySeeds[ i ] );
      }
      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     true,
                                     false,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag::value >::SubentityContainerType          SubentityContainerType;
   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag::value >::SharedContainerType             SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                                    EntityInitializerType;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                               EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

		IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
		for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
			subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );

		BaseType::initSubentities(entity, entityIndex, entitySeed, meshInitializer);
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     false,
                                     false,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >                BaseType;

   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag::value >::SubentityContainerType        SubentityContainerType;
   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag::value >::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::DataType                                            GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                   InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                                  EntityInitializerType;
   typedef tnlDimensionsTag< EntityTopology::dimensions >                                      EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                           EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
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
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag,
                                     false,
                                     false,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTopology,
                                           typename DimensionsTag::Decrement >
{};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     tnlDimensionsTag< 0 >,
                                     true,
                                     false,
                                     true >
{
   typedef tnlDimensionsTag< 0 >                                  DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag::value >                 SubentitiesTraits;

   typedef typename SubentitiesTraits::AccessArrayType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                           InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >          EntityInitializerType;
   typedef tnlDimensionsTag< EntityTopology::dimensions >              EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << std::endl;
		   const IdArrayType &subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >(entity);
		   for( LocalIndexType i = 0; i < subentityIdsArray.getSize(); i++ )
			   meshInitializer.template getSuperentityInitializer< DimensionsTag >().addSuperentity( EntityDimensionsTag(), subentityIdsArray[ i ], entityIndex);
	}

};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     tnlDimensionsTag< 0 >,
                                     true,
                                     false,
                                     false >
{
   typedef tnlMeshInitializer< MeshConfig >         InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig,
                                     EntityTopology >   EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTopology, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
 
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};
 
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     tnlDimensionsTag< 0 >,
                                     false,
                                     true,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< MeshConfig >                  InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&, InitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology,
          bool SuperEntityStorage >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTopology,
                                     tnlDimensionsTag< 0 >,
                                     false,
                                     false,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< MeshConfig >                  InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTopology,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                                                    EntityType;

   protected:
   void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&,
                         InitializerType& ) {}
};

} // namespace TNL
