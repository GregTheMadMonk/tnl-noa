/***************************************************************************
                          tnlMeshEntityInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHENTITYINITIALIZER_H_
#define TNLMESHENTITYINITIALIZER_H_

#include <core/tnlStaticFor.h>
#include <mesh/initializer/tnlMeshSuperentityStorageInitializer.h>
#include <mesh/initializer/tnlMeshSubentitySeedCreator.h>

template< typename MeshConfig >
class tnlMeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SubentityStorage = tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled,
          bool SubentityOrientationStorage = tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::orientationEnabled,
          bool SuperentityStorage = tnlMeshSuperentityTraits< MeshConfig,
                                                              typename tnlMeshSubentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::SubentityTag,
                                                              EntityTopology::dimensions >::storageEnabled >
class tnlMeshEntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTag >
class tnlMeshEntityInitializer
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag, 
                                           tnlDimensionsTag< EntityTag::dimensions - 1 > >
{
   typedef tnlDimensionsTag< EntityTag::dimensions >                                 DimensionsTag;
   private:

      typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag, 
                                           tnlDimensionsTag< EntityTag::dimensions - 1 > > BaseType;
      
   typedef
      tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     tnlDimensionsTag< EntityTag::dimensions - 1 > >   SubentityBaseType;
   typedef
      tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename
                                          tnlMeshTraits< MeshConfig >::DimensionsTag > SuperentityBaseType;

   typedef typename tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::EntityType               EntityType;   
   typedef typename tnlMeshEntityTraits< MeshConfig,
                                           DimensionsTag::value >::GlobalIndexType      GlobalIndexType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTag, 0 >                         SubvertexTag;
   typedef typename SubvertexTag::ContainerType::ElementType                                 VertexGlobalIndexType;
   typedef typename SubvertexTag::ContainerType::IndexType                                   VertexLocalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                   InitializerType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                            SeedType;

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
   typename tnlMeshSuperentityTraits< MeshConfig, EntityTag, SuperentityDimensionTag::value >::SharedContainerType& getSuperentityContainer( SuperentityDimensionTag )
   {
      return this->entity->template getSuperentitiesIndices< SuperentityDimensionTag::value >();
   }

   static void setEntityVertex( EntityType& entity,
                                VertexLocalIndexType localIndex,
                                VertexGlobalIndexType globalIndex )
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
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     true,
                                     false,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTag, DimensionsTag::value >             SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
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
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     true,
                                     true,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTag, DimensionsTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag::value >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityIdsArray[ i ] = subentityIndex;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         //cout << "    Subentity orientation = " << subentityOrientationsArray[ i ].getSubvertexPermutation() << endl;
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
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     true,
                                     true,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentityTraits< MeshConfig, EntityTag, DimensionsTag::value >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         //cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );         
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentitySeeds[ i ] ).createOrientation( subentitySeeds[ i ] );
      }
      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};


template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     true,
                                     false,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTag,
                                              DimensionsTag::value >::SubentityContainerType          SubentityContainerType;
   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTag,
                                              DimensionsTag::value >::SharedContainerType             SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                                    EntityInitializerType;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                               EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

		IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
		for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
			subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );

		BaseType::initSubentities(entity, entityIndex, entitySeed, meshInitializer);
   }
};

template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     false,
                                     false,
                                     true >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                BaseType;

   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTag,
                                              DimensionsTag::value >::SubentityContainerType        SubentityContainerType;
   typedef typename tnlMeshSubentityTraits< MeshConfig,
                                              EntityTag,
                                              DimensionsTag::value >::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::DataType                                            GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                                                   InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                                  EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                      EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                           EntityType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
      
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         //cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
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
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     DimensionsTag,
                                     false,
                                     false,
                                     false >
   : public tnlMeshEntityInitializerLayer< MeshConfig,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{};

template< typename MeshConfig,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     true,
                                     false,
                                     true >
{
   typedef tnlDimensionsTag< 0 >                                  DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value >                 SubentitiesTraits;

   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;

   typedef tnlMeshInitializer< MeshConfig >                           InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >          EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >              EntityDimensionsTag;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag::value >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
      
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};

};

template< typename MeshConfig,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     true,
                                     false,
                                     false >
{
   typedef tnlMeshInitializer< MeshConfig >         InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig,
                                     EntityTag >   EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< MeshConfig, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                             LocalIndexType;


   protected:
   
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};
   
};

template< typename MeshConfig,
          typename EntityTag,
          bool SuperEntityStorage >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     false,
                                     true,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< MeshConfig >                  InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&, InitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTag,
          bool SuperEntityStorage >
class tnlMeshEntityInitializerLayer< MeshConfig,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     false,
                                     false,
                                     SuperEntityStorage > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< MeshConfig >                  InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentityTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< MeshConfig, EntityTag >                                                    EntityType;

   protected:
   void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&,
                         InitializerType& ) {}
};


#endif /* TNLMESHENTITYINITIALIZER_H_ */
