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
         typename DimensionsTag,
         typename SubentityStorageTag = typename tnlMeshSubentitiesTraits< ConfigTag,
                                                                           EntityTag,
                                                                           DimensionsTag >::SubentityStorageTag,
         typename SuperentityStorageTag = typename tnlMeshSuperentitiesTraits< ConfigTag,
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
                                           tnlDimensionsTag< EntityTag::dimensions - 1 > >,
     public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename tnlMeshTraits< ConfigTag >::DimensionsTag >
{
   typedef tnlDimensionsTag< EntityTag::dimensions >                                 DimensionsTag;
   private:

   typedef
      tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< EntityTag::dimensions - 1 > >   SubentityBaseType;
   typedef
      tnlMeshSuperentityInitializerLayer< ConfigTag,
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

   template< typename > class SubentitiesCreator;

   public:

   //using SuperentityBaseType::setNumberOfSuperentities;

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

      this->entity->setId( entityIndex );

      initSuperentities();
      initSubentities( meshInitializer );
      //cout << " Entity initiation done ... " << endl;
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
   typename tnlMeshSubentitiesTraits< ConfigTag, EntityTag, SubentityDimensionTag >::SharedContainerType& subentityContainer( SubentityDimensionTag )
   {
      return this->entity->template getSubentitiesIndices< SubentityDimensionTag::value >();
   }
   
   template< typename SuperentityDimensionTag >
   bool setNumberOfSuperentities( SuperentityDimensionTag, typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, SuperentityDimensionTag >::ContainerType::IndexType size )
   {
      return this->entity->template setNumberOfSuperentities< SuperentityDimensionTag::value >( size );
   }

   // TODO: check if we need the following two methods
   /*template< typename SuperentityDimensionTag >
   bool getSuperentityContainer( SuperentityDimensionTag, typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, SuperentityDimensionTag >::ContainerType::IndexType size )
   {
      return this->entity->template setNumberOfSuperentities< SuperentityDimensionTag::value >( size );
   }*/

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

   void initSubentities( InitializerType& meshInitializer )
   {
      //cout << "   Initiating subentities of entity ... " << endl;
      SubentityBaseType::initSubentities( *this, meshInitializer );
   }

   void initSuperentities()
   {
      //cout << "   Initiating superentities..." << endl;
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
         tnlMeshSubentitiesTraits< ConfigTag,
                                   EntityTag,
                                   SubentityDimensionTag >::SharedContainerType            SubentitiesIndicesContainerType;

      typedef typename tnlMeshSubentitiesTraits< ConfigTag,
                                                 EntityTag,
                                                 SubentityDimensionTag>::SubentityContainerType
                                                                                           SubentityContainerType;

      enum { subentitiesCount       = Tag::count };
      enum { subentityVerticesCount = tnlMeshSubentitiesTraits< ConfigTag,
                                                                SubentityTag,
                                                                tnlDimensionsTag< 0 > >::count };

      public:
      static void createSubentities( SubentityContainerType& subentities,
                                     const EntityType &entity )
      {
         const SubentitiesIndicesContainerType& subvertexIndices = entity.template getSubentitiesIndices< 0 >();
         //cout << "        entity = " << entity << endl;
         //cout << "        subvertexIndices = " << subvertexIndices << endl;
         tnlStaticFor< LocalIndexType, 0, subentitiesCount, CreateSubentities >::exec( subentities, subvertexIndices );
      }

      private:
      template< LocalIndexType subentityIndex >
      class CreateSubentities
      {
         public:
         static void exec( SubentityContainerType &subentities,
                           const SubentitiesIndicesContainerType& subvertexIndices )
         {
            SubentityType &subentity = subentities[ subentityIndex ];
            tnlStaticFor< LocalIndexType, 0, subentityVerticesCount, SetSubentityVertex >::exec( subentity, subvertexIndices );
         }

         private:
         template< LocalIndexType subentityVertexIndex >
         class SetSubentityVertex
         {
            public:
            static bool exec( SubentityType &subentity,
                              const SubentitiesIndicesContainerType& subvertexIndices )
            {
               LocalIndexType vertexIndex = Tag::template Vertex< subentityIndex, subentityVertexIndex >::index;
               //cout << "        Setting subentity " << subentityIndex << " vertex " << subentityVertexIndex << " to " << subvertexIndices[ vertexIndex ] << endl;
               tnlMeshEntityInitializer< ConfigTag, SubentityTag >::setEntityVertex( subentity, subentityVertexIndex, subvertexIndices[ vertexIndex ] );
               return true;
            }
         };
      };
   };
};

template< typename ConfigTag >
class tnlMeshEntityInitializer< ConfigTag, tnlMeshVertexTag >
   : public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                tnlMeshVertexTag,
                                                typename tnlMeshTraits< ConfigTag >::DimensionsTag >
{
   typedef tnlDimensionsTag< 0 >                                                                     DimensionsTag;

   typedef tnlMeshSuperentityInitializerLayer< ConfigTag,
                                               tnlMeshVertexTag,
                                               typename tnlMeshTraits< ConfigTag >::DimensionsTag >     SuperentityBaseType;

   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::Type                          EntityType;
   typedef typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::ContainerType::IndexType      GlobalIndexType;

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
      this->entity->setId( this->entityIndex );
      initSuperentities();
      //cout << "Vertex initiation done ... " << endl;
   }

   template< typename SuperentityDimensionsTag >
   bool setNumberOfSuperentities( SuperentityDimensionsTag,
                                  const typename EntityType::template SuperentitiesTraits< SuperentityDimensionsTag::value >::ContainerType::IndexType size )
   {
      return this->entity->template setNumberOfSuperentities< SuperentityDimensionsTag::value >( size );
   }

   template< typename SuperentityDimensionsTag >
   typename tnlMeshSuperentitiesTraits< ConfigTag,
                                        tnlMeshVertexTag,
                                        SuperentityDimensionsTag >::SharedContainerType&
      getSuperentityContainer( SuperentityDimensionsTag )
   {
      return this->entity->template getSuperentitiesIndices< SuperentityDimensionsTag::value >();
   }

   private:

   EntityType *entity;
   GlobalIndexType entityIndex;

   void initSuperentities()
   {
      //cout << "    Initiating superentities of vertex ..." << endl;
      SuperentityBaseType::initSuperentities(*this);
   }
};

/****
 * Mesh entity initializer layer
 */

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
                                           typename DimensionsTag::Decrement >
{
   typedef tnlMeshEntityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >                   BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag, EntityTag, DimensionsTag >                     SubentitiesTraits;
   typedef typename SubentitiesTraits::SubentityContainerType                                  SubentityContainerType;
   typedef typename SubentitiesTraits::SharedContainerType                                     SharedContainerType;
   typedef typename SharedContainerType::ElementType                                           GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                    EntityInitializerType;
   typedef tnlDimensionsTag< EntityTag::dimensions >                                        EntityDimensionsTag;

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      //cout << "      Initiating subentities with " << DimensionsTag::value << " dimensions..." << endl;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );
      SharedContainerType& subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      //cout << "      Subentities = " << subentities << endl;
      for( typename SubentityContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntityIndex( subentities[ i ] );
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         subentityContainer[ i ] = subentityIndex;
         //cout << "       Setting " << i << "-th subentity to " << subentityContainer[ i ] << endl;
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
                                     tnlStorageTraits< true >,
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

   typedef tnlMeshInitializer< ConfigTag >                                                     InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                                    EntityInitializerType;

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      //cout << "      Initiating subentities with " << DimensionsTag::value << " dimensions..." << endl;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );
      SharedContainerType& subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      for( typename SubentityContainerType::IndexType i = 0;
           i < subentityContainer.getSize();
           i++ )
      {
         subentityContainer[ i ] = meshInitializer.findEntityIndex( subentities[ i ] );
         //cout << "       Setting " << i << "-th subentity to " << subentityContainer[ i ] << endl;
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

   protected:
   void initSubentities( EntityInitializerType& entityInitializer,
                         InitializerType& meshInitializer )
   {
      SubentityContainerType subentities;
      //cout << "      Initiating subentities with " << DimensionsTag::value << " dimensions..." << endl;
      entityInitializer.template createSubentities< DimensionsTag >( subentities );

      for( typename SubentityContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         GlobalIndexType subentityIndex = meshInitializer.findEntityIndex( subentities[ i ] );
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         //cout << "       NOT setting " << i << "-th subentity to " << subentityIndex << endl;
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
                                           typename DimensionsTag::Decrement >
{};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
                                     tnlStorageTraits< true >,
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

   protected:
   void initSubentities( EntityInitializerType &entityInitializer,
                         InitializerType &meshInitializer )
   {
      //cout << "      Initiating subentities with " << DimensionsTag::value << " dimensions..." << endl;
      const SharedContainerType &subentityContainer = entityInitializer.subentityContainer( DimensionsTag() );
      for (typename SharedContainerType::IndexType i = 0; i < subentityContainer.getSize(); i++)
      {
         GlobalIndexType subentityIndex = subentityContainer[ i ];
         //cout << "       Setting " << i << "-th subentity to " << subentityContainer[ i ] << endl;
         tnlAssert( subentityIndex >= 0,
                   cerr << " subentityContainer = " << subentityContainer << endl; );
         GlobalIndexType superentityIndex = entityInitializer.getEntityIndex();
         tnlAssert( superentityIndex >= 0, );
         meshInitializer.getEntityInitializer( DimensionsTag(), subentityIndex).addSuperentity( EntityDimensionsTag(), superentityIndex );
      }
   }
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializerLayer< ConfigTag,
                                     EntityTag,
                                     tnlDimensionsTag< 0 >,
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
                                     tnlDimensionsTag< 0 >,
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
                                     tnlDimensionsTag< 0 >,
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
