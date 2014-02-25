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
          typename EntityTag,
          typename DimensionsTraits,
          typename SuperentityStorageTag = typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                                                EntityTag,
                                                                                DimensionsTraits >::SuperentityStorageTag >
class tnlMeshSuperentityInitializerLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializer
   : public tnlMeshEntityInitializerLayer< ConfigTag,
                                           EntityTag, tnlDimensionsTraits< EntityTag::dimensions - 1 > >,
     public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{
   typedef tnlDimensionsTraits< EntityTag::dimensions >                                 DimensionsTraits;

   typedef
      tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTraits< EntityTag::dimension - 1> >   SubentityBaseType;
   typedef
      tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename
                                             tnlMeshTraits< ConfigTag >::DimensionsTraits > SuperentityBaseType;

   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >::Type               EntityType;
   typedef typename tnlMeshEntitiesTraits< ConfigTag,
                                           DimensionsTraits >::ContainerType::IndexType      GlobalIndexType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, tnlDimensionsTraits< 0 > >        SubvertexTag;
   typedef typename SubvertexTag::ContainerType::DataType                                    VertexGlobalIndexType;
   typedef typename SubvertexTag::ContainerType::IndexType                                   VertexLocalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                   InitializerType;

   template< typename > class SubentitiesCreator;

   public:

   tnlMeshEntityInitializer() : entity(0), entityIndex( -1 ) {}

   void init( EntityType& entity, GlobalIndexType entityIndex )
   {
      entity = &entity;
      entityIndex = entityIndex;
   }

   void initEntity( InitializerType &meshInitializer )
   {
      tnlAssert( entity, );

      entity->setID( entityIndex );

      initSuperentities();
      initSubentities( meshInitializer );
   }

   template< typename SubentityDimensionsTag >
   void createSubentities( typename tnlMeshSubentitiesTraits< ConfigTag,
                                                              EntityTag,
                                                              SubentityDimensionsTag>::SubentityContainerType& subentities) const
   {
      SubentitiesCreator< SubentityDimensionsTag >::createSubentities( subentities, *entity);
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
      typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, SubentityDimensionTag >   Tag;
      typedef typename Tag::SubentityTag                                                SubentityTag;
      typedef typename Tag::SubentityType                                               SubentityType;
      typedef typename Tag::ContainerType::IndexType                                    LocalIndexType;

      typedef typename EntityType::SubentityIndicesArrayType                            SubentityIndicesArrayType;
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



#endif /* TNLMESHENTITYINITIALIZER_H_ */
