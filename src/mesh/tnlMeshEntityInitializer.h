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
#include <mesh/tnlMeshSuperentityInitializerLayer.h>

template< typename ConfigTag >
class tnlMeshInitializer;

template<typename ConfigTag,
         typename EntityTag,
         typename DimensionsTraits,
         typename SubentityStorageTag = typename tnlMeshSubentitiesTraits< ConfigTag,
                                                                           EntityTag,
                                                                           DimensionsTraits >::SubentityStorageTag,
         typename SuperentityStorageTag = typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                                               typename tnlMeshSubentitiesTraits< ConfigTag,
                                                                                                                  EntityTag,
                                                                                                                  DimensionsTraits >::SubentityTag,
                                                                               tnlDimensionsTraits< EntityTag::dimensions > >::SuperentityStorageTag >
class tnlMeshEntityInitializerLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializer
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag, 
                                           tnlDimensionsTraits< EntityTag::dimensions - 1 > >,
     public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{
   typedef tnlDimensionsTraits< EntityTag::dimensions >                                 DimensionsTraits;

   typedef
      tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< EntityTag::dimensions - 1> >   SubentityBaseType;
   typedef
      tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename
                                             tnlMeshTraits< ConfigTag >::DimensionsTraits > SuperentityBaseType;

   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >::Type               EntityType;
   typedef typename tnlMeshEntitiesTraits< ConfigTag,
                                           DimensionsTraits >::ContainerType::IndexType      GlobalIndexType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, tnlDimensionsTraits< 0 > >        SubvertexTag;
   typedef typename SubvertexTag::ContainerType::ElementType                                 VertexGlobalIndexType;
   typedef typename SubvertexTag::ContainerType::IndexType                                   VertexLocalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                   InitializerType;

   template< typename > class SubentitiesCreator;

   public:

   static tnlString getType() {};

   tnlMeshEntityInitializer() : entity(0), entityIndex( -1 ) {}

   void init( EntityType& entity, GlobalIndexType entityIndex )
   {
      this->entity = &entity;
      this->entityIndex = entityIndex;
   }

   void initEntity( InitializerType &meshInitializer )
   {
      tnlAssert( this->entity, );

      this->entity->setID( entityIndex );

      initSuperentities();
      initSubentities( meshInitializer );
   }

   template< typename SubentityDimensionsTag >
   void createSubentities( typename tnlMeshSubentitiesTraits< ConfigTag,
                                                              EntityTag,
                                                              SubentityDimensionsTag >::SubentityContainerType& subentities ) const
   {
      SubentitiesCreator< SubentityDimensionsTag >::createSubentities( subentities, *entity );
   }

   GlobalIndexType getEntityIndex() const
   {
      tnlAssert( entityIndex >= 0,
                 cerr << "entityIndex = " << entityIndex );
      return this->entityIndex;
   }

   template< typename SubentityDimensionTag >
   typename tnlMeshSubentitiesTraits< ConfigTag, EntityTag, SubentityDimensionTag >::ContainerType& subentityContainer( SubentityDimensionTag )
   {
      return this->entity->subentityIndicesContainer(SubentityDimensionTag());
   }

   template< typename SuperentityDimensionTag >
   typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, SuperentityDimensionTag >::ContainerType& superentityContainer( SuperentityDimensionTag )
   {
      return this->entity->superentityIndicesContainer( SuperentityDimensionTag() );
   }

   static void setEntityVertex( EntityType& entity,
                                VertexLocalIndexType localIndex,
                                VertexGlobalIndexType globalIndex )
   {
      entity.setVertex( localIndex, globalIndex );
   }

   private:
   EntityType *entity;
   GlobalIndexType entityIndex;

   void initSubentities( InitializerType& meshInitializer )
   {
      SubentityBaseType::initSubentities( *this, meshInitializer );
   }

   void initSuperentities()
   {
      SuperentityBaseType::initSuperentities( *this) ;
   }

   template< typename SubentityDimensionTag >
   class SubentitiesCreator
   {
      typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, SubentityDimensionTag >     Tag;
      typedef typename Tag::SubentityTag                                                  SubentityTag;
      typedef typename Tag::SubentityType                                                 SubentityType;
      typedef typename Tag::ContainerType::IndexType                                      LocalIndexType;

      typedef typename
         EntityType::template SubentitiesTraits< DimensionsTraits::value >::ContainerType SubentityIndicesArrayType;
      typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                                 EntityTag,
                                                 SubentityDimensionTag>::SubentityContainerType
                                                                                           SubentityContainerType;

      enum { subentitiesCount       = Tag::count };
      enum { subentityVerticesCount = tnlMeshSubentitiesTraits< ConfigTag,
                                                                SubentityTag,
                                                                tnlDimensionsTraits< 0 > >::count };

      public:
      static void createSubentities( SubentityContainerType& subentities,
                                     const EntityType &entity )
      {
         SubentityIndicesArrayType subvertexIndices = entity.template subentityIndices< 0 >();
         tnlStaticFor< LocalIndexType, 0, subentitiesCount, CreateSubentities>::Exec( subentities, subvertexIndices );
      }

   private:
      template<LocalIndexType subentityIndex>
      class CreateSubentities
      {
      public:
         static void exec(SubentityContainerType &subentities, SubentityIndicesArrayType subvertexIndices)
         {
            SubentityType &subentity = subentities[subentityIndex];
            tnlStaticFor<LocalIndexType, 0, subentityVerticesCount, SetSubentityVertex>::Exec(subentity, subvertexIndices);
         }

      private:
         template<LocalIndexType subentityVertexIndex>
         class SetSubentityVertex
         {
         public:
            static void exec(SubentityType &subentity, SubentityIndicesArrayType subvertexIndices)
            {
               LocalIndexType vertexIndex = Tag::template Vertex< subentityIndex, subentityVertexIndex >::index;
               tnlMeshEntityInitializer< ConfigTag, SubentityTag >::setEntityVertex( subentity, subentityVertexIndex, subvertexIndices[vertexIndex] );
            }
         };
      };
   };
};

template< typename ConfigTag >
class tnlMeshEntityInitializer< ConfigTag, tnlMeshVertexTag >
   : public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                tnlMeshVertexTag,
                                                typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{
   typedef tnlDimensionsTraits< 0 >                                                                     DimensionsTraits;

   typedef tnlMeshSuperentityInitializerLayer< ConfigTag,
                                               tnlMeshVertexTag,
                                               typename tnlMeshTraits< ConfigTag >::DimensionsTraits >     SuperentityBaseType;

   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >::Type                          EntityType;
   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >::ContainerType::IndexType      GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                              InitializerType;

   public:

   tnlMeshEntityInitializer() : entity(0), entityIndex(-1) {}

   static tnlString getType() {};

   void init( EntityType& entity, GlobalIndexType entityIndex )
   {
      this->entity = &entity;
      this->entityIndex = entityIndex;
   }

   void initEntity(InitializerType &meshInitializer)
   {
      this->entity->setID( this->entityIndex );
      initSuperentities();
   }

   template< typename SuperentityDimensionsTag >
   typename tnlMeshSuperentitiesTraits< ConfigTag,
                                        tnlMeshVertexTag,
                                        SuperentityDimensionsTag >::ContainerType&
      getSuperentityContainer( SuperentityDimensionsTag )
   {
      return this->entity->superentityIndicesContainer( SuperentityDimensionsTag() );
   }

   private:

   EntityType *entity;
   GlobalIndexType entityIndex;

   void initSuperentities()
   {
      SuperentityBaseType::initSuperentities(*this);
   }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< true > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Previous >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Previous >                   BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, DimensionsTag >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::ContainerType                                           ContainerType;
   typedef typename ContainerType::ElementType                                                 GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                    EntityInitializerType;
   typedef tnlDimensionsTraits< EntityTag::dimensions >                                        EntityDimensionsTraits;

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );

      ContainerType& subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      for( typename SubentityContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntityIndex( subentities[ i ] );
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         subentityContainer[ i ] = subentityIndex;
         meshInitializer.getEntityInitializer( DimensionsTag(), subentityIndex ).addSuperentity( EntityDimensionsTraits(), superentityIndex );
      }

      BaseType::initSubentities( entityInitializer, meshInitializer );
   }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Previous >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Previous >                   BaseType;

   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SubentityContainerType              SubentityContainerType;
   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::ContainerType                       ContainerType;

   typedef tnlMeshInitializer< ConfigTag >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                    EntityInitializerType;

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );

      ContainerType& subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      for( typename SubentityContainerType::IndexType i = 0;
           i < subentityContainer.getSize();
           i++ )
         subentityContainer[ i ] = meshInitializer.findEntityIndex( subentities[ i ] );

      BaseType::initSubentities( entityInitializer, meshInitializer );
   }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Previous >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Previous >                BaseType;

   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::SubentityContainerType        SubentityContainerType;
   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::ContainerType                 ContainerType;
   typedef typename ContainerType::DataType                                                  GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                   InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                  EntityInitializerType;
   typedef tnlDimensionsTraits< EntityTag::dimensions >                                      EntityDimensionsTag;

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );

      for( typename SubentityContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntityIndex( subentities[ i ] );
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         meshInitializer.getEntityInitializer( DimensionsTag(), subentityIndex ).addSuperentity( EntityDimensionsTag(), superentityIndex );
      }

      BaseType::initSubentities( entityInitializer, meshInitializer );
   }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     DimensionsTag,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > >
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag,
                                           typename DimensionsTag::Previous >
{};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< 0 >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< true > >
{
   typedef tnlDimensionsTraits< 0 >                                                   DimensionsTag;

   typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                              EntityTag,
                                              DimensionsTag >::ContainerType          ContainerType;
   typedef typename ContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                            InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                           EntityInitializerType;
   typedef tnlDimensionsTraits< EntityTag::dimensions >                                EntityDimensionsTag;

protected:
   void initSubentities(EntityInitializerType &entityInitializer, InitializerType &meshInitializer)
   {
      const ContainerType &subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      for (typename ContainerType::IndexType i = 0; i < subentityContainer.getSize(); i++)
      {
         GlobalIndexType subentityIndex = subentityContainer[i];
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         meshInitializer.getEntityInitializer( DimensionsTag(), subentityIndex).addSuperentity( EntityDimensionsTag(), superentityIndex );
      }
   }
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< 0 >,
                                     tnlStorageTraits< true >,
                                     tnlStorageTraits< false > >
{
   typedef tnlMeshInitializer< ConfigTag >         InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     EntityTag >   EntityInitializerType;

   protected:
   
   void initSubentities( EntityInitializerType&, InitializerType& ) {}
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< 0 >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< true > > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< ConfigTag >                  InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag > EntityInitializerType;

   protected:
   void initSubentities( EntityInitializerType&, InitializerType& ) {}
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< 0 >,
                                     tnlStorageTraits< false >,
                                     tnlStorageTraits< false > > // Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
{
   typedef tnlMeshInitializer< ConfigTag >                  InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag > EntityInitializerType;

   protected:
   void initSubentities( EntityInitializerType&,
                         InitializerType& ) {}
};


#endif /* TNLMESHENTITYINITIALIZER_H_ */
