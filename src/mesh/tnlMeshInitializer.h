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

   bool initMesh( MeshType& mesh )
   {
      //cout << "======= Starting mesh initiation ========" << endl;
      this->setMesh( mesh );
      if( ! this->checkCells() )
         return false;
      //cout << "========= Creating entities =============" << endl;
      this->createEntitiesFromCells();
      this->createEntityInitializers();
      //cout << "====== Initiating entities ==============" << endl;
      this->initEntities( *this );
      //cout << "Mesh initiation done..." << endl;
      return true;
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

   bool checkCells()
   {
      typedef typename tnlMeshEntity< ConfigTag, EntityTag >::template SubentitiesTraits< 0 >::LocalIndexType LocalIndexType;
      for( GlobalIndexType cell = 0;
           cell < this->getMesh().getNumberOfCells();
           cell++ )
         for( LocalIndexType i = 0;
              i < this->getMesh().getCell( cell ).getNumberOfVertices();
              i++ )
            if( this->getMesh().getCell( cell ).getVerticesIndices()[ i ] == - 1 )
            {
               cerr << "The cell number " << cell << " does not have properly set vertex index number " << i << "." << endl;
               return false;
            }
      return true;
   }

   void createEntitiesFromCells()
   {
      //cout << " Creating entities with " << DimensionsTraits::value << " dimensions..." << endl;
      cellInitializerContainer.setSize( this->getMesh().getNumberOfCells() );
      for( GlobalIndexType cell = 0;
           cell < this->getMesh().getNumberOfCells();
           cell++ )
      {
         //cout << "  Creating the cell number " << cell << endl;
         CellInitializerType& cellInitializer = cellInitializerContainer[ cell ];

         cellInitializer.init( this->getMesh().getCell( cell ), cell );
         BaseType::createEntitiesFromCells( cellInitializer );
      }

   }

   void initEntities( InitializerType& meshInitializer )
   {
      //cout << " Initiating entities with " << DimensionsTraits::value << " dimensions..." << endl;
      for( typename CellInitializerContainerType::IndexType i = 0;
           i < cellInitializerContainer.getSize();
           i++ )
      {
         //cout << "  Initiating entity " << i << " with " << DimensionsTraits::value << " dimensions..." << endl;
         cellInitializerContainer[ i ].initEntity( meshInitializer );
      }
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
      bool entityFound = uniqueContainer.find( entity, idx );
      tnlAssert( entityFound,
                 cerr << " entity = " << entity << endl; );
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
      //cout << " Creating entities with " << DimensionsTraits::value << " dimensions..." << endl;
      SubentitiesContainerType subentities;
      cellInitializer.template createSubentities< DimensionsTraits >( subentities );
      for( typename SubentitiesContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         //cout << "      Inserting subentity " << endl << subentities[ i ] << endl;
         uniqueContainer.insert( subentities[ i ] );
      }
      //cout << " Container with entities with " << DimensionsTraits::value << " dimensions has: " << endl << this->uniqueContainer << endl;
      BaseType::createEntitiesFromCells( cellInitializer );
   }

   void createEntityInitializers()
   {
      entityInitializerContainer.setSize( uniqueContainer.getSize() );
      BaseType::createEntityInitializers();
   }

   void initEntities( InitializerType &meshInitializer )
   {
      //cout << " Initiating entities with " << DimensionsTraits::value << " dimensions..." << endl;
      //cout << " Container with entities with " << DimensionsTraits::value << " dimensions has: " << endl << this->uniqueContainer << endl;
      const GlobalIndexType numberOfEntities = uniqueContainer.getSize();
      this->getMesh().template setNumberOfEntities< DimensionsTraits::value >( numberOfEntities );
      uniqueContainer.toArray( this->getMesh().template getEntities< DimensionsTraits::value >() );
      uniqueContainer.reset();
      //cout << "  this->getMesh().template getEntities< DimensionsTraits::value >() has: " << this->getMesh().template getEntities< DimensionsTraits::value >() << endl;

      //ContainerType& entityContainer = this->getMesh().entityContainer(DimensionsTraits());
      for( GlobalIndexType i = 0;
           i < numberOfEntities;
           i++)
      {
         //cout << "Initiating entity " << i << " with " << DimensionsTraits::value << " dimensions..." << endl;
         EntityInitializerType& entityInitializer = entityInitializerContainer[ i ];
         //cout << "Initiating with entity " << this->getMesh().template getEntity< DimensionsTraits::value >( i ) << endl;
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
      tnlAssert( index >= 0 && index < vertexInitializerContainer.getSize(),
               cerr << " index = " << index
                    << " vertexInitializerContainer.getSize() = " << vertexInitializerContainer.getSize() << endl; );
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
      //cout << " Initiating entities with " << DimensionsTraits::value << " dimensions..." << endl;
      SharedContainerType& vertexContainer = this->getMesh().template getEntities< DimensionsTraits::value >();
      for( GlobalIndexType i = 0;
           i < vertexContainer.getSize();
           i++ )
      {
         //cout << "Initiating entity " << i << " with " << DimensionsTraits::value << " dimensions..." << endl;
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
