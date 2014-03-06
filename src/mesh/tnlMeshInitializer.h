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
#include <mesh/traits/tnlStorageTraits.h>

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

template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               typename tnlMeshTraits< ConfigTag >::DimensionsTraits,
                               tnlStorageTraits< true > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename tnlMeshTraits< ConfigTag >::DimensionsTraits::Previous >
{
   typedef typename tnlMeshTraits< ConfigTag >::DimensionsTraits        DimensionsTraits;

   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTraits::Previous >   BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >         Tag;
   typedef typename Tag::Tag                                            EntityTag;
   typedef typename Tag::ContainerType                                  ContainerType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;

   typedef tnlMeshInitializer< ConfigTag >                              InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >             CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >    CellInitializerContainerType;

   public:
   using BaseType::getEntityInitializer;
   CellInitializerType& getEntityInitializer( DimensionsTraits, GlobalIndexType index )
   {
      return cellInitializerContainer[ index ];
   }

   protected:
   void createEntitiesFromCells()
   {
      cout << "Creating entities with dimensions " << DimensionsTraits::value << endl;
      //ContainerType& cellContainer = this->getMesh().entityContainer( DimensionsTraits());

      //cellInitializerContainer.create( cellContainer.getSize());
      cellInitializerContainer.setSize( this->getMesh().getNumberOfCells() );
      for( GlobalIndexType cell = 0;
           cell < this->getMesh().getNumberOfCells();
           cell++ )
      {
         cout << "Setting the cell number " << cell << endl;
         CellInitializerType& cellInitializer = cellInitializerContainer[ cell ];
         cellInitializer.init( this->getMesh().getCell( cell ), cell );
         BaseType::createEntitiesFromCells( cellInitializer );
      }

   }

   void initEntities( InitializerType& meshInitializer )
   {
      for( typename CellInitializerContainerType::IndexType i = 0;
           i < cellInitializerContainer.getSize();
           i++ )
         cellInitializerContainer[ i ].initEntity( meshInitializer );

      cellInitializerContainer.reset();

      BaseType::initEntities( meshInitializer );
   }

   private:
   CellInitializerContainerType cellInitializerContainer;
};


template< typename ConfigTag,
          typename DimensionsTraits >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTraits,
                               tnlStorageTraits< true > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTraits::Previous >
{
   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTraits::Previous >  BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >            Tag;
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
      GlobalIndexType idx;
      uniqueContainer.find( entity, idx );
      return idx;
   }

   using BaseType::getEntityInitializer;
   EntityInitializerType& getEntityInitializer( DimensionsTraits, GlobalIndexType index )
   {
      return entityInitializerContainer[ index ];
   }

   protected:

   void createEntitiesFromCells( const CellInitializerType& cellInitializer )
   {
      SubentitiesContainerType subentities;
      cellInitializer.template createSubentities< DimensionsTraits >( subentities );
      for( typename SubentitiesContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
         uniqueContainer.insert( subentities[ i ] );

      BaseType::createEntitiesFromCells( cellInitializer );
   }

   void createEntityInitializers()
   {
      entityInitializerContainer.setSize( uniqueContainer.getSize() );

      BaseType::createEntityInitializers();
   }

   void initEntities(InitializerType &meshInitializer)
   {
      const GlobalIndexType numberOfEntities = uniqueContainer.getSize();
      this->getMesh().template setNumberOfEntities< DimensionsTraits::value >( numberOfEntities );
      cout << " DimensionsTraits::value = " << DimensionsTraits::value << endl;
      //uniqueContainer.toArray( this->getMesh().template getEntities< DimensionsTraits::value >() );
      uniqueContainer.reset();

      //ContainerType& entityContainer = this->getMesh().entityContainer(DimensionsTraits());
      for( GlobalIndexType i = 0;
           i < numberOfEntities;
           i++)
      {
         EntityInitializerType& entityInitializer = entityInitializerContainer[ i ];
         entityInitializer.init( this->getMesh().template getEntity< DimensionsTraits::value >( i ), i );
         entityInitializer.initEntity( meshInitializer );
      }

      entityInitializerContainer.reset();

      BaseType::initEntities( meshInitializer );
   }

   private:
   UniqueContainerType uniqueContainer;
   EntityInitializerContainerType entityInitializerContainer;
};

template< typename ConfigTag,
          typename DimensionsTraits >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTraits,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTraits::Previous >
{};


template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               tnlDimensionsTraits< 0 >,
                               tnlStorageTraits< true > >
{
   typedef tnlMesh< ConfigTag >                                        MeshType;
   typedef tnlDimensionsTraits< 0 >                                    DimensionsTraits;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >        Tag;
   typedef typename Tag::Tag                                           EntityTag;
   typedef typename Tag::ContainerType                                 ContainerType;
   typedef typename Tag::SharedContainerType                           SharedContainerType;
   typedef typename ContainerType::IndexType                           GlobalIndexType;

   typedef typename tnlMeshTraits< ConfigTag >::CellType               CellType;

   typedef tnlMeshInitializer< ConfigTag >                             InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, 
                                     typename ConfigTag::CellTag >     CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >            VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType > VertexInitializerContainerType;

   public:

   void setMesh( MeshType& mesh )
   {
      this->mesh = &mesh;
   }

   MeshType& getMesh()
   {
      tnlAssert( this->mesh, );
      return *( this->mesh );
   }

   VertexInitializerType& getEntityInitializer( DimensionsTraits, GlobalIndexType index )
   {
      return vertexInitializerContainer[ index ];
   }

   protected:
   void findEntityIndex() const                               {} // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
   void createEntitiesFromCells( const CellInitializerType& ) {}

   void createEntityInitializers()
   {
      vertexInitializerContainer.setSize( this->getMesh().template getNumberOfEntities< DimensionsTraits::value >() );
   }

   void initEntities( InitializerType& meshInitializer )
   {
      SharedContainerType& vertexContainer = this->getMesh().template getEntities< DimensionsTraits::value >();
      for( GlobalIndexType i = 0;
           i < vertexContainer.getSize();
           i++ )
      {
         VertexInitializerType& vertexInitializer = vertexInitializerContainer[ i ];
         vertexInitializer.init( vertexContainer[ i ], i );
         vertexInitializer.initEntity( meshInitializer );
      }

      vertexInitializerContainer.reset();
   }

   private:
   VertexInitializerContainerType vertexInitializerContainer;

   MeshType* mesh;
};




#endif /* TNLMESHINITIALIZER_H_ */
