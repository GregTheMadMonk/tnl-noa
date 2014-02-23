/***************************************************************************
                          tnlMeshInitializer.h  -  description
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

#ifndef TNLMESHINITIALIZER_H_
#define TNLMESHINITIALIZER_H_

#include <mesh/traits/tnlMeshEntitiesTraits.h>
#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>
#include <mesh/tnlMeshEntityInitializer.h>
#include <mesh/tnlMesh.h>

template< typename ConfigTag,
          typename DimensionsTraits,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< ConfigTag,
                                                                      DimensionsTraits >::EntityStorageTag >
class tnlMeshInitializerLayer;


template< typename ConfigTag,
          typename EntityTag>
class tnlMeshEntityInitializer;

template< typename ConfigTag >
class tnlMeshInitializer
        : public tnlMeshInitializerLayer< ConfigTag,
                                          typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{
   typedef tnlMesh< ConfigTag > MeshType;

   public:

   void initMesh( MeshType& mesh )
   {
      this->setMesh( mesh );
      this->createEntitiesFromCells();
      this->createEntityInitializers();
      this->initEntities( *this );
   }
};

template< typename ConfigTag,
          typename DimensionsTraits >
class tnmlMeshInitializerLayer< ConfigTag,
                                DimensionsTraits, tnlMeshStorageTraits< true > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTraits::Previous >
{
   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTraits::Previous >  BaseType;

   typedef tnlMeshEntitiesTag< ConfigTag, DimensionsTraits >               Tag;
   typedef typename Tag::Tag                                               EntityTag;
   typedef typename Tag::Type                                              EntityType;
   typedef typename Tag::ContainerType                                     ContainerType;
   typedef typename Tag::UniqueContainerType                               UniqueContainerType;
   typedef typename ContainerType::IndexType                               GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                                 InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     typename ConfigTag::CellTag >         CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;

   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                typename ConfigTag::CellTag,
                                DimensionsTraits >::SubentityContainerType SubentitiesContainerType;

   public:

   using BaseType::findEntityIndex;
   GlobalIndexType findEntityIndex( EntityType &entity ) const
   {
      return uniqueContainer.find( entity );
   }

   using BaseType::getEntityInitializer;
   EntityInitializerType& getEntityInitializer( DimensionsTraits, GlobalIndexType index )
   {
      return entityInitializerContainer[ index ];
   }

   protected:

   void createEntitiesFromCells( const CellInitializerType& cellInitializer )
   {
      SubentitiesContainerType subntities;
      cellInitializer.template createSubentities< DimensionsTraits>( subentities );

      for( typename SubentitiesContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
         uniqueContainer.insert( subentities[ i ] );

      BaseType::createEntitiesFromCells(cellInitializer);
   }

   void createEntityInitializers()
   {
      m_entityInitializerContainer.create(m_uniqueContainer.getSize());

      BaseType::createEntityInitializers();
   }

   void initEntities(InitializerType &meshInitializer)
   {
      this->getMesh().entityContainer(DimensionTag()).create(m_uniqueContainer.getSize());
      m_uniqueContainer.toArray(this->getMesh().entityContainer(DimensionTag()));
      m_uniqueContainer.free();

      ContainerType &entityContainer = this->getMesh().entityContainer(DimensionTag());
      for (GlobalIndexType i = 0; i < entityContainer.getSize(); i++)
      {
         EntityInitializerType &entityInitializer = m_entityInitializerContainer[i];
         entityInitializer.init(entityContainer[i], i);
         entityInitializer.initEntity(meshInitializer);
      }

      m_entityInitializerContainer.free();

      BaseType::initEntities(meshInitializer);
   }

private:
   UniqueContainerType m_uniqueContainer;
   EntityInitializerContainerType m_entityInitializerContainer;
};


#endif /* TNLMESHINITIALIZER_H_ */
