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
#include <mesh/tnlMeshSuperentityStorageInitializer.h>
#include <mesh/tnlMeshSubentitySeedCreator.h>

template< typename ConfigTag >
class tnlMeshInitializer;

template<typename ConfigTag,
         typename EntityTag,
         typename DimensionsTag,
         typename SubentityStorageTag =
            typename tnlMeshSubentitiesTraits< ConfigTag,
                                               EntityTag,
                                               DimensionsTag >::SubentityStorageTag,
         typename SubentityOrientationStorage = 
            tnlStorageTraits< tnlMeshConfigTraits< ConfigTag >::
               template SubentityTraits< EntityTag, DimensionsTag >::orientationEnabled >,
         typename SuperentityStorageTag = 
            typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                 typename tnlMeshSubentitiesTraits< ConfigTag,
                                                                                    EntityTag,
                                                                                    DimensionsTag >::SubentityTag,
                                                                               tnlDimensionsTag< EntityTag::dimensions > >::SuperentityStorageTag >
class tnlMeshEntityInitializerLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializer
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag, 
                                           tnlDimensionsTag< EntityTag::dimensions - 1 > >
{
   typedef tnlDimensionsTag< EntityTag::dimensions >                                 DimensionsTag;
   private:

      typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag, 
                                           tnlDimensionsTag< EntityTag::dimensions - 1 > > BaseType;
      
   typedef
      tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< EntityTag::dimensions - 1 > >   SubentityBaseType;
   typedef
      tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename
                                          tnlMeshTraits< ConfigTag >::DimensionsTag > SuperentityBaseType;

   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::Type               EntityType;
   typedef typename tnlMeshEntitiesTraits< ConfigTag,
                                           DimensionsTag >::ContainerType::IndexType      GlobalIndexType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, tnlDimensionsTag< 0 > >        SubvertexTag;
   typedef typename SubvertexTag::ContainerType::ElementType                                 VertexGlobalIndexType;
   typedef typename SubvertexTag::ContainerType::IndexType                                   VertexLocalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                   InitializerType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                            SeedType;

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
   typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, SuperentityDimensionTag >::SharedContainerType& getSuperentityContainer( SuperentityDimensionTag )
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
class tnlMeshEntityInitializer< MeshConfig, tnlMeshVertexTag >
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
template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, DimensionsTag >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
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
template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< true > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, DimensionsTag >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
         GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
         subentityIdsArray[ i ] = subentityIndex;
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentityIndex ).createOrientation( subentitySeeds[ i ] );
         cout << "    Subentity orientation = " << subentityOrientationsArray[ i ].getSubvertexPermutation() << endl;
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
template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, DimensionsTag >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                          InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                         EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                                EntityDimensionsTag;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                                SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                      SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;
   typedef typename SubentitiesTraits::OrientationArrayType                                    OrientationArrayType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

      IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
      OrientationArrayType &subentityOrientationsArray = InitializerType::template subentityOrientationsArray< DimensionsTag >( entity );
      for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
      {         
         cout << "    Adding subentity " << subentityIdsArray[ i ] << endl;
         subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );         
         subentityOrientationsArray[ i ] = meshInitializer.template getReferenceOrientation< DimensionsTag >( subentitySeeds[ i ] ).createOrientation( subentitySeeds[ i ] );
      }
      BaseType::initSubentities( entity, entityIndex, entitySeed, meshInitializer );
   }
};


template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SubentityContainerType          SubentityContainerType;
   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SharedContainerType             SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                    EntityInitializerType;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                               EntityType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                InitializerType& meshInitializer )
   {
      cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
      auto subentitySeeds = SubentitySeedsCreatorType::create( entitySeed );

		IdArrayType& subentityIdsArray = InitializerType::template subentityIdsArray< DimensionsTag >( entity );
		for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++)
			subentityIdsArray[ i ] = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );

		BaseType::initSubentities(entity, entityIndex, entitySeed, meshInitializer);
   }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                BaseType;

   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SubentityContainerType        SubentityContainerType;
   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::DataType                                            GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                   InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                  EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                      EntityDimensionsTag;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                           EntityType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;


   protected:
      
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer )
      {
         cout << "   Initiating subentities with " << DimensionsTag::value << " dimensions ... " << endl;
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

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Decrement >
{};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > >
{
   typedef tnlDimensionsTag< 0 >                                  DimensionsTag;
   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTag >                 SubentitiesTraits;

   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                           InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >          EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >              EntityDimensionsTag;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;
      typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;


   protected:
      
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};

};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > >
{
   typedef tnlMeshInitializer< ConfigTag >         InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     EntityTag >   EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTag >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;
      typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                                           SeedType;
   typedef tnlMeshSubentitySeedsCreator< ConfigTag, EntityTag, DimensionsTag >                 SubentitySeedsCreatorType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< EntityTag, DimensionsTag >::IdArrayType IdArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::LocalIndexType                                             LocalIndexType;


   protected:
   
      static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, const SeedType& entitySeed,
                                   InitializerType& meshInitializer ) {};
   
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< ConfigTag >                  InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTag >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;

   protected:
   static void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&, InitializerType& ) {}
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< ConfigTag >                  InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag > EntityInitializerType;
   typedef tnlDimensionsTag< 0 >                   DimensionsTag;
   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTag >                 SubentitiesTraits;
   typedef typename SubentitiesTraits::SharedContainerType           SharedContainerType;
   typedef typename SharedContainerType::ElementType                 GlobalIndexType;
   typedef tnlMeshEntity< ConfigTag, EntityTag >                                                    EntityType;

   protected:
   void initSubentities( EntityType& entity, GlobalIndexType entityIndex, EntityInitializerType&,
                         InitializerType& ) {}
};


#endif /* TNLMESHENTITYINITIALIZER_H_ */
