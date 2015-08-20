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
#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>
#include <mesh/tnlMeshEntityInitializer.h>
#include <mesh/tnlMesh.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/tnlMeshSubentitySeedCreator.h>

template< typename ConfigTag,
          typename DimensionsTag,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< ConfigTag,
                                                                      DimensionsTag >::EntityStorageTag >
class tnlMeshInitializerLayer;


template< typename ConfigTag,
          typename EntityTag>
class tnlMeshEntityInitializer;

template< typename ConfigTag >
class tnlMeshInitializer
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename tnlMeshTraits< ConfigTag >::DimensionsTag >
{
   typedef tnlMesh< ConfigTag > MeshType;
   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename tnlMeshTraits< ConfigTag >::DimensionsTag > BaseType;


   public:

   tnlMeshInitializer()
   : verbose( false ), mesh( 0 )
   {}

   void setVerbose( bool verbose )
   {
      this->verbose = verbose;
   }

   typedef typename tnlMeshConfigTraits< ConfigTag >::PointArrayType    PointArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   
   bool createMesh( const PointArrayType& points,
                    const CellSeedArrayType& cellSeeds,
                    MeshType& mesh )   
   {      
      cout << "======= Starting mesh initiation ========" << endl;
      this->mesh = &mesh;
      
      cout << "========= Creating entity seeds =============" << endl;
      BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
		//BaseType::createEntityReferenceOrientations();
      cout << "====== Initiating entities ==============" << endl;
      //BaseType::initEntities(*this, points, cellSeeds);
      
      return true;
   }

   template< typename SuperDimensionsTag, typename MeshEntity>
	static typename tnlMeshConfigTraits< ConfigTag >::IdArrayAccessorType& superentityIdsArray( MeshEntity& entity )
	{
		return entity.template superentityIdsArray< SuperDimensionsTag >();
	}
   
   template< typename DimensionsTag >
	typename tnlMeshConfigTraits< ConfigTag >::template EntityTraits< DimensionsTag>::ContainerType& meshEntitiesArray()
   {
      return mesh->template entitiesArray< DimensionsTag >();
   }
   
   template< typename DimensionsTag, typename SuperDimensionsTag >
	typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType& meshSuperentityIdsArray()
   {
      return mesh->template superentityIdsArray< DimensionsTag, SuperDimensionsTag >();
   }
   
   protected:

   bool verbose;
   
   MeshType* mesh;
};

/****
 * Mesh initializer layer for cells
 */
template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               typename tnlMeshTraits< ConfigTag >::DimensionsTag,
                               tnlStorageTraits< true > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename tnlMeshTraits< ConfigTag >::DimensionsTag::Decrement >
{
   typedef typename tnlMeshTraits< ConfigTag >::DimensionsTag        DimensionsTag;

   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTag::Decrement >   BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >         Tag;
   typedef typename Tag::Tag                                            EntityTag;
   typedef typename Tag::ContainerType                                  ContainerType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellType          CellType;

   typedef tnlMeshInitializer< ConfigTag >                              InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >             CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >    CellInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;

   public:
   using BaseType::getEntityInitializer;
   CellInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
   {
      //return cellInitializerContainer[ index ];
   }

   protected:
      
   void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
   {
      /*typedef tnlMeshSubentitySeedsCreator< ConfigTag, CellType, DimensionsTag >  SubentitySeedsCreator;
      for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )         
      {
         //typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
         //SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
         SubentitySeedsCreator::create( cellSeeds[ i ], this->seedsIndexedSet );
      }*/
      BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
   }
      

   bool checkCells()
   {
      typedef typename tnlMeshEntity< ConfigTag, EntityTag >::template SubentitiesTraits< 0 >::LocalIndexType LocalIndexType;
      const GlobalIndexType numberOfVertices( this->getMesh().getNumberOfVertices() );
      for( GlobalIndexType cell = 0;
           cell < this->getMesh().getNumberOfCells();
           cell++ )
         for( LocalIndexType i = 0;
              i < this->getMesh().getCell( cell ).getNumberOfVertices();
              i++ )
         {
            if( this->getMesh().getCell( cell ).getVerticesIndices()[ i ] == - 1 )
            {
               cerr << "The cell number " << cell << " does not have properly set vertex index number " << i << "." << endl;
               return false;
            }
            if( this->getMesh().getCell( cell ).getVerticesIndices()[ i ] >= numberOfVertices )
            {
               cerr << "The cell number " << cell << " does not have properly set vertex index number " << i
                    << ". The index " << this->getMesh().getCell( cell ).getVerticesIndices()[ i ]
                    << "is higher than the number of all vertices ( " << numberOfVertices
                    << " )." << endl;
               return false;
            }
         }
      return true;
   }

   void createEntitiesFromCells( bool verbose )
   {
      //cout << " Creating entities with " << DimensionsTag::value << " dimensions..." << endl;
      /*cellInitializerContainer.setSize( this->getMesh().getNumberOfCells() );
      for( GlobalIndexType cell = 0;
           cell < this->getMesh().getNumberOfCells();
           cell++ )
      {
         if( verbose )
            cout << "  Creating the cell number " << cell << "            \r " << flush;
         CellInitializerType& cellInitializer = cellInitializerContainer[ cell ];

         //cellInitializer.init( this->getMesh().getCell( cell ), cell );
         BaseType::createEntitiesFromCells( cellInitializer );
      }
      if( verbose )
         cout << endl;*/

   }

   void initEntities( InitializerType& meshInitializer )
   {
      /*cout << " Initiating cells..." << endl;
      for( typename CellInitializerContainerType::IndexType i = 0;
           i < cellInitializerContainer.getSize();
           i++ )
      {
         //cout << "  Initiating entity " << i << " with " << DimensionsTag::value << " dimensions..." << endl;
         //cellInitializerContainer[ i ].initEntity( meshInitializer );
      }
      cellInitializerContainer.reset();
      //cout << "Initiating superentities ...." << endl;
      //superentityInitializer.initSuperentities( meshInitializer );
      BaseType::initEntities( meshInitializer );*/
   }

   private:
      typedef  typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;
      typedef  tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                   EntityTag,
                                                   typename tnlMeshTraits< ConfigTag >::DimensionsTag >  SuperentityInitializerType;

      //CellInitializerContainerType cellInitializerContainer;
      //SuperentityInitializer superentityInitializer;

      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 */
template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTag,
                               tnlStorageTraits< true > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTag::Decrement >
{
   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTag::Decrement >  BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >            Tag;
   typedef typename Tag::Tag                                               EntityTag;
   typedef typename Tag::Type                                              EntityType;
   typedef typename Tag::ContainerType                                     ContainerType;
   typedef typename Tag::UniqueContainerType                               UniqueContainerType;
   typedef typename ContainerType::IndexType                               GlobalIndexType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellType          CellType;

   typedef tnlMeshInitializer< ConfigTag >                                 InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     typename ConfigTag::CellType >         CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;

   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                typename ConfigTag::CellType,
                                DimensionsTag >::SubentityContainerType SubentitiesContainerType;

   public:

   using BaseType::findEntityIndex;
   GlobalIndexType findEntityIndex( EntityType &entity ) const
   {
      /*GlobalIndexType idx;
      bool entityFound = uniqueContainer.find( entity, idx );
      tnlAssert( entityFound,
                 cerr << " entity = " << entity << endl; );
      return idx;*/
   }

   using BaseType::getEntityInitializer;
   EntityInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
   {
      //return entityInitializerContainer[ index ];
   }

   protected:
   
   void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
   {
      typedef tnlMeshSubentitySeedsCreator< ConfigTag, CellType, DimensionsTag >  SubentitySeedsCreator;
      for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )         
      {
         //SubentitySeedsCreator::create( cellSeeds[ i ], this->seedsIndexedSet );
         typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
         SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
         for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
         {
            tnlMeshEntitySeed< tnlMeshConfigBase< CellType >, EntityTag >& entitySeed = subentytiSeeds[ j ];
            this->seedsIndexedSet.insert( subentytiSeeds[ j ] );
         }
      }
      BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
   }

   void createEntitiesFromCells( const CellInitializerType& cellInitializer )
   {
     /* //cout << " Creating entities with " << DimensionsTag::value << " dimensions..." << endl;
      SubentitiesContainerType subentities;
      cellInitializer.template createSubentities< DimensionsTag >( subentities );
      for( typename SubentitiesContainerType::IndexType i = 0;
           i < subentities.getSize();
           i++ )
      {
         //cout << "      Inserting subentity " << endl << subentities[ i ] << endl;
         uniqueContainer.insert( subentities[ i ] );
      }
      //cout << " Container with entities with " << DimensionsTag::value << " dimensions has: " << endl << this->uniqueContainer << endl;
      BaseType::createEntitiesFromCells( cellInitializer );
      */
   }

   void createEntityInitializers()
   {
      //entityInitializerContainer.setSize( uniqueContainer.getSize() );
      //BaseType::createEntityInitializers();
   }

   void initEntities( InitializerType &meshInitializer )
   {
      /* cout << " Initiating entities with " << DimensionsTag::value << " dimensions..." << endl;
      //cout << " Container with entities with " << DimensionsTag::value << " dimensions has: " << endl << this->uniqueContainer << endl;
      const GlobalIndexType numberOfEntities = uniqueContainer.getSize();
      this->getMesh().template setNumberOfEntities< DimensionsTag::value >( numberOfEntities );
      uniqueContainer.toArray( this->getMesh().template getEntities< DimensionsTag::value >() );
      uniqueContainer.reset();
      //cout << "  this->getMesh().template getEntities< DimensionsTag::value >() has: " << this->getMesh().template getEntities< DimensionsTag::value >() << endl;

      //ContainerType& entityContainer = this->getMesh().entityContainer(DimensionsTag());
      for( GlobalIndexType i = 0;
           i < numberOfEntities;
           i++)
      {
         //cout << "Initiating entity " << i << " with " << DimensionsTag::value << " dimensions..." << endl;
         EntityInitializerType& entityInitializer = entityInitializerContainer[ i ];
         //cout << "Initiating with entity " << this->getMesh().template getEntity< DimensionsTag::value >( i ) << endl;
         //entityInitializer.init( this->getMesh().template getEntity< DimensionsTag::value >( i ), i );
         //entityInitializer.initEntity( meshInitializer );
      }

      entityInitializerContainer.reset();
      cout << "Initiating superentities for entities with " << DimensionsTag::value << " dimensions ..." << endl;
      cout << "Storage is " << tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, typename tnlMeshTraits< ConfigTag >::DimensionsTag >::SuperentityStorageTag::enabled << endl;
      superentityInitializer.initSuperentities( meshInitializer );
      BaseType::initEntities( meshInitializer );
      */
   }

   private:
      /*typedef  tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                   EntityTag,
                                                   typename tnlMeshTraits< ConfigTag >::DimensionsTag >  SuperentityInitializer;
      
      SuperentityInitializer superentityInitializer;
      UniqueContainerType uniqueContainer;
      EntityInitializerContainerType entityInitializerContainer;*/
      
      typedef  typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;
      typedef  tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                   EntityTag,
                                                   typename tnlMeshTraits< ConfigTag >::DimensionsTag >  SuperentityInitializerType;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTag,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTag::Decrement >
{};

/****
 * Mesh initializer layer for vertices
 */
/*template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               tnlDimensionsTag< tnlMeshConfigTraits< ConfigTag >::meshDimensions  >,
                               tnlStorageTraits< true > > :
   public tnlMeshInitializerLayer< ConfigTag, typename tnlDimensionsTag< tnlMeshConfigTraits< ConfigTag >::meshDimensions >::Decrement >
{
   typedef tnlMeshInitializerLayer< ConfigTag, typename tnlDimensionsTag< tnlMeshConfigTraits< ConfigTag >::meshDimensions >::Decrement > BaseType;
   
   typedef tnlMesh< ConfigTag >                                        MeshType;
   typedef tnlDimensionsTag< tnlMeshConfigTraits< ConfigTag >::meshDimensions  >    DimensionsTag;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >        Tag;
   typedef typename Tag::Tag                                           EntityTag;
   typedef typename Tag::ContainerType                                 ContainerType;
   typedef typename Tag::SharedContainerType                           SharedContainerType;
   typedef typename ContainerType::IndexType                           GlobalIndexType;

   typedef typename tnlMeshTraits< ConfigTag >::CellType               CellType;

   typedef tnlMeshInitializer< ConfigTag >                             InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, 
                                     typename ConfigTag::CellType >     CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >            VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType > VertexInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;

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

   VertexInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
   {
      tnlAssert( index >= 0 && index < vertexInitializerContainer.getSize(),
               cerr << " index = " << index
                    << " vertexInitializerContainer.getSize() = " << vertexInitializerContainer.getSize() << endl; );
      return vertexInitializerContainer[ index ];
   }

   protected:
      
   void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
   {
      BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
   };
      
   void findEntityIndex() const                               {} // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
   void createEntitiesFromCells( const CellInitializerType& ) {}

   void createEntityInitializers()
   {
      vertexInitializerContainer.setSize( this->getMesh().template getNumberOfEntities< DimensionsTag::value >() );
   }

   void initEntities( InitializerType& meshInitializer )
   {
      //cout << " Initiating entities with " << DimensionsTag::value << " dimensions..." << endl;
      SharedContainerType& vertexContainer = this->getMesh().template getEntities< DimensionsTag::value >();
      for( GlobalIndexType i = 0;
           i < vertexContainer.getSize();
           i++ )
      {
         //cout << "Initiating entity " << i << " with " << DimensionsTag::value << " dimensions..." << endl;
         VertexInitializerType& vertexInitializer = vertexInitializerContainer[ i ];
         //vertexInitializer.init( vertexContainer[ i ], i );
         //vertexInitializer.initEntity( meshInitializer );
      }
      cout << "Initiating superentities for entities with " << DimensionsTag::value << " dimensions ..." << endl;
      cout << "Storage is " << tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, typename tnlMeshTraits< ConfigTag >::DimensionsTag >::SuperentityStorageTag::enabled << endl;
      superentityInitializer.initSuperentities( meshInitializer );
      vertexInitializerContainer.reset();
   }

   private:
      typedef  tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                   EntityTag,
                                                   typename tnlMeshTraits< ConfigTag >::DimensionsTag >  SuperentityInitializer;
      
      SuperentityInitializer superentityInitializer;

      VertexInitializerContainerType vertexInitializerContainer;

      MeshType* mesh;
};*/


/****
 * Mesh initializer layer for vertices
 */
template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               tnlDimensionsTag< 0 >,
                               tnlStorageTraits< true > >
{
   typedef tnlMesh< ConfigTag >                                        MeshType;
   typedef tnlDimensionsTag< 0 >                                    DimensionsTag;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >        Tag;
   typedef typename Tag::Tag                                           EntityTag;
   typedef typename Tag::ContainerType                                 ContainerType;
   typedef typename Tag::SharedContainerType                           SharedContainerType;
   typedef typename ContainerType::IndexType                           GlobalIndexType;

   typedef typename tnlMeshTraits< ConfigTag >::CellType               CellType;

   typedef tnlMeshInitializer< ConfigTag >                             InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, 
                                     typename ConfigTag::CellType >     CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >            VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType > VertexInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;

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

   VertexInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
   {
      tnlAssert( index >= 0 && index < vertexInitializerContainer.getSize(),
               cerr << " index = " << index
                    << " vertexInitializerContainer.getSize() = " << vertexInitializerContainer.getSize() << endl; );
      return vertexInitializerContainer[ index ];
   }

   protected:
      
   void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds ){};
      
   void findEntityIndex() const                               {} // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
   void createEntitiesFromCells( const CellInitializerType& ) {}

   void createEntityInitializers()
   {
      vertexInitializerContainer.setSize( this->getMesh().template getNumberOfEntities< DimensionsTag::value >() );
   }

   void initEntities( InitializerType& meshInitializer )
   {
      //cout << " Initiating entities with " << DimensionsTag::value << " dimensions..." << endl;
      SharedContainerType& vertexContainer = this->getMesh().template getEntities< DimensionsTag::value >();
      for( GlobalIndexType i = 0;
           i < vertexContainer.getSize();
           i++ )
      {
         //cout << "Initiating entity " << i << " with " << DimensionsTag::value << " dimensions..." << endl;
         VertexInitializerType& vertexInitializer = vertexInitializerContainer[ i ];
         //vertexInitializer.init( vertexContainer[ i ], i );
         //vertexInitializer.initEntity( meshInitializer );
      }
      cout << "Initiating superentities for entities with " << DimensionsTag::value << " dimensions ..." << endl;
      cout << "Storage is " << tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, typename tnlMeshTraits< ConfigTag >::DimensionsTag >::SuperentityStorageTag::enabled << endl;
      superentityInitializer.initSuperentities( meshInitializer );
      vertexInitializerContainer.reset();
   }

   private:
      typedef  tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                   EntityTag,
                                                   typename tnlMeshTraits< ConfigTag >::DimensionsTag >  SuperentityInitializer;
      
      SuperentityInitializer superentityInitializer;

      VertexInitializerContainerType vertexInitializerContainer;

      MeshType* mesh;
};




#endif /* TNLMESHINITIALIZER_H_ */
