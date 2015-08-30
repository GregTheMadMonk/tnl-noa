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

#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlMeshEntitiesTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>
#include <mesh/tnlMeshEntityInitializer.h>
#include <mesh/tnlMesh.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/tnlMeshSubentitySeedCreator.h>
#include <mesh/tnlMeshSuperentityStorageInitializer.h>
#include <mesh/tnlMeshEntityReferenceOrientation.h>
#include <mesh/tnlMeshEntitySeed.h>
#include <mesh/tnlMeshEntitySeedKey.h>

template< typename MeshConfig,
          typename DimensionsTag,
          typename EntityStorageTag = 
             typename tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >::EntityStorageTag,
          typename EntityReferenceOrientationStorage = 
             tnlStorageTraits< tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::orientationNeeded > >
class tnlMeshInitializerLayer;


template< typename MeshConfig,
          typename EntityTag>
class tnlMeshEntityInitializer;

template< typename MeshConfig >
class tnlMeshInitializer
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename tnlMeshTraits< MeshConfig >::DimensionsTag >
{
   typedef tnlMesh< MeshConfig > MeshType;
   typedef tnlMeshInitializerLayer< MeshConfig,
                                    typename tnlMeshTraits< MeshConfig >::DimensionsTag > BaseType;


   public:

   tnlMeshInitializer()
   : verbose( false ), mesh( 0 )
   {}

   void setVerbose( bool verbose )
   {
      this->verbose = verbose;
   }

   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType    PointArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType CellSeedArrayType;
   
   bool createMesh( const PointArrayType& points,
                    const CellSeedArrayType& cellSeeds,
                    MeshType& mesh )   
   {      
      cout << "======= Starting mesh initiation ========" << endl;
      this->mesh = &mesh;
      
      cout << "========= Creating entity seeds =============" << endl;
      BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
		
      cout << "========= Creating entity reference orientations =============" << endl;
      BaseType::createEntityReferenceOrientations();
      
      cout << "====== Initiating entities ==============" << endl;
      BaseType::initEntities( *this, points, cellSeeds );
      
      return true;
   }
   
   template<typename SubDimensionsTag, typename EntityType >
   static typename tnlMeshTraits< MeshConfig >::template SubentityTraits< typename EntityType::Tag, SubDimensionsTag::value >::IdArrayType&
   subentityIdsArray( EntityType& entity )
   {
      return entity.template subentityIdsArray< SubDimensionsTag >();
   }
   
   template< typename SuperDimensionsTag, typename MeshEntity>
   static typename tnlMeshTraits< MeshConfig >::IdArrayAccessorType&
   superentityIdsArray( MeshEntity& entity )
   {
      return entity.template superentityIdsArray< SuperDimensionsTag >();
   }
   
   template<typename SubDimensionsTag, typename MeshEntity >
	static typename tnlMeshTraits< MeshConfig >::template SubentityTraits< typename MeshEntity::Tag, SubDimensionsTag::value >::OrientationArrayType&
   subentityOrientationsArray( MeshEntity &entity )
   {
      return entity.template subentityOrientationsArray< SubDimensionsTag >();
   }
   
   template< typename DimensionsTag >
   typename tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::StorageArrayType&
   meshEntitiesArray()
   {
      return mesh->template entitiesArray< DimensionsTag >();
   }
   
   template< typename DimensionsTag, typename SuperDimensionsTag >
	typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType&
   meshSuperentityIdsArray()
   {
      return mesh->template superentityIdsArray< DimensionsTag, SuperDimensionsTag >();
   }
   
   static void
   setVertexPoint( typename MeshType::VertexType& vertex, const typename MeshType::PointType& point )
   {
      vertex.setPoint( point );
   }
   
   template< typename DimensionsTag >
   tnlMeshSuperentityStorageInitializer< MeshConfig, typename tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::Tag >&
   getSuperentityInitializer()
   {
      return BaseType::getSuperentityInitializer( DimensionsTag() );
   }

   typedef typename tnlMeshTraits< MeshConfig >::GlobalIndexType GlobalIndexType;
   template< typename DimensionsTag >
	const tnlMeshEntityReferenceOrientation< MeshConfig, typename tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::Tag >&
   getReferenceOrientation( GlobalIndexType index) const
	{
		return BaseType::getReferenceOrientation( DimensionsTag(), index);
	}

   protected:

   bool verbose;
   
   MeshType* mesh;
};

/****
 * Mesh initializer layer for cells
 *  - entities storage must turned on (cells must always be stored )
 *  - entities orientation does not make sense for cells => it is turned off
 */
template< typename MeshConfig >
class tnlMeshInitializerLayer< MeshConfig,
                               typename tnlMeshTraits< MeshConfig >::DimensionsTag,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename tnlMeshTraits< MeshConfig >::DimensionsTag::Decrement >
{
   typedef typename tnlMeshTraits< MeshConfig >::DimensionsTag        DimensionsTag;

   typedef tnlMeshInitializerLayer< MeshConfig,
                                    typename DimensionsTag::Decrement >   BaseType;

   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            EntityTraits;
   typedef typename EntityTraits::Tag                                            EntityTag;
   typedef typename EntityTraits::StorageArrayType                                  ContainerType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::CellTopology      CellTopology;
   typedef typename EntityTraits::StorageArrayType                          EntityArrayType;

   typedef tnlMeshInitializer< MeshConfig >                              InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >             EntityInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >             CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >    CellInitializerContainerType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType    LocalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, CellTopology >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTag >  SuperentityInitializerType;

   public:
      using BaseType::getEntityInitializer;
      CellInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         //return cellInitializerContainer[ index ];
      }

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      void initEntities( InitializerType &initializer, const PointArrayType &points, const CellSeedArrayType &cellSeeds)
      {
         EntityArrayType &entityArray = initializer.template meshEntitiesArray< DimensionsTag >();
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << endl;
         entityArray.setSize( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < entityArray.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << endl;
            EntityInitializerType::initEntity( entityArray[i], i, cellSeeds[i], initializer );
         }
         /***
          * There are no superentities in this layer storing mesh cells.
          */
         
         BaseType::initEntities( initializer, points );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed ) const
      {
         return this->seedsIndexedSet.find( seed );
      }

      using BaseType::getSuperentityInitializer;
      SuperentityInitializerType& getSuperentityInitializer( DimensionsTag )
      {
         return this->superentityInitializer;
      }
   
      bool checkCells()
      {
         typedef typename tnlMeshEntity< MeshConfig, EntityTag >::template SubentitiesTraits< 0 >::LocalIndexType LocalIndexType;
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

   private:
      typedef  typename tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;      

      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned off
 */
template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshInitializerLayer< MeshConfig,
                               DimensionsTag,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
   typedef tnlMeshInitializerLayer< MeshConfig,
                                    typename DimensionsTag::Decrement >  BaseType;

   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            Tag;
   typedef typename Tag::Tag                                               EntityTag;
   typedef typename Tag::Type                                              EntityType;
   typedef typename Tag::ContainerType                                     ContainerType;
   typedef typename Tag::UniqueContainerType                               UniqueContainerType;
   typedef typename ContainerType::IndexType                               GlobalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::CellTopology          CellTopology;

   typedef tnlMeshInitializer< MeshConfig >                                 InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig,
                                     typename MeshConfig::CellTopology >         CellInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType    CellSeedArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType       LocalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            EntityTraits;
   typedef typename EntityTraits::ContainerType                          EntityArrayType;
   typedef typename EntityTraits::SeedArrayType                          SeedArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTag >  SuperentityInitializerType;


   typedef typename
      tnlMeshSubentitiesTraits< MeshConfig,
                                typename MeshConfig::CellTopology,
                                DimensionsTag::value >::SubentityContainerType SubentitiesContainerType;

   public:

      using BaseType::getEntityInitializer;
      EntityInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         //return entityInitializerContainer[ index ];
      }

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef tnlMeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionsTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )         
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << endl;
               //tnlMeshEntitySeed< tnlMeshConfigBase< CellTopology >, EntityTag >& entitySeed = subentytiSeeds[ j ];
               this->seedsIndexedSet.insert( subentytiSeeds[ j ] );
            }
         }
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed ) const
      {
         GlobalIndexType index;
         this->seedsIndexedSet.find( seed, index );
         return index;
      }
      
      using BaseType::getSuperentityInitializer;
      SuperentityInitializerType& getSuperentityInitializer( DimensionsTag )
      {
         return this->superentityInitializer;
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points )
      {
         EntityArrayType &entityArray = initializer.template meshEntitiesArray< DimensionsTag >();
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << endl;
         entityArray.setSize( this->seedsIndexedSet.getSize() );
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << endl;
            EntityInitializerType::initEntity( entityArray[ i ], i, seedsArray[ i ], initializer );
         }
         this->seedsIndexedSet.reset();

         this->superentityInitializer.initSuperentities( initializer );

         BaseType::initEntities(initializer, points);
      }

      void createEntityReferenceOrientations() const {}
   private:
      
      typedef  typename tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned on
 */
template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshInitializerLayer< MeshConfig,
                               DimensionsTag,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< true > >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
   typedef tnlMeshInitializerLayer< MeshConfig,
                                    typename DimensionsTag::Decrement >  BaseType;

   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            Tag;
   typedef typename Tag::Tag                                               EntityTag;
   typedef typename Tag::EntityType                                              EntityType;
   typedef typename Tag::StorageArrayType                                  ContainerType;
   typedef typename Tag::UniqueContainerType                               UniqueContainerType;
   typedef typename ContainerType::IndexType                               GlobalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::CellTopology          CellTopology;

   typedef tnlMeshInitializer< MeshConfig >                                 InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig,
                                     typename MeshConfig::CellTopology >         CellInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType    CellSeedArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType       LocalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            EntityTraits;
   typedef typename EntityTraits::StorageArrayType                          EntityArrayType;
   typedef typename EntityTraits::SeedArrayType                          SeedArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTag >  SuperentityInitializerType;
   typedef typename EntityTraits::ReferenceOrientationType               ReferenceOrientationType;
   typedef typename EntityTraits::ReferenceOrientationArrayType          ReferenceOrientationArrayType;


   typedef typename
      tnlMeshSubentitiesTraits< MeshConfig,
                                typename MeshConfig::CellTopology,
                                DimensionsTag::value >::SubentityContainerType SubentitiesContainerType;

   public:      
      
      using BaseType::getEntityInitializer;
      EntityInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         //return entityInitializerContainer[ index ];
      }

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef tnlMeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionsTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )         
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << endl;
               //tnlMeshEntitySeed< tnlMeshConfigBase< CellTopology >, EntityTag >& entitySeed = subentytiSeeds[ j ];
               this->seedsIndexedSet.insert( subentytiSeeds[ j ] );
            }
         }
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed ) const
      {
         GlobalIndexType index;
         this->seedsIndexedSet.find( seed, index );
         return index;
      }
      
      using BaseType::getSuperentityInitializer;
      SuperentityInitializerType& getSuperentityInitializer( DimensionsTag )
      {
         return this->superentityInitializer;
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points )
      {
         EntityArrayType &entityArray = initializer.template meshEntitiesArray< DimensionsTag >();
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << endl;
         entityArray.setSize( this->seedsIndexedSet.getSize() );
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << endl;
            EntityInitializerType::initEntity( entityArray[ i ], i, seedsArray[ i ], initializer );
         }
         this->seedsIndexedSet.reset();

         this->superentityInitializer.initSuperentities( initializer );

         BaseType::initEntities(initializer, points);
      }

      using BaseType::getReferenceOrientation;
      const ReferenceOrientationType& getReferenceOrientation( DimensionsTag, GlobalIndexType index) const
      {
         return this->referenceOrientations[ index ];
      }
      
      void createEntityReferenceOrientations()
      {
         //cout << " Creating entity reference orientations with " << DimensionsTag::value << " dimensions ... " << endl;
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         this->referenceOrientations.setSize( seedsArray.getSize() );
         for( GlobalIndexType i = 0; i < seedsArray.getSize(); i++ )
         {
            //cout << "  Creating reference orientation for entity " << i << endl;
            this->referenceOrientations[ i ] = ReferenceOrientationType( seedsArray[ i ] );
         }
         BaseType::createEntityReferenceOrientations();
		}	
      
   private:
      
      typedef  typename tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
      ReferenceOrientationArrayType referenceOrientations;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshInitializerLayer< MeshConfig,
                               DimensionsTag,
                               tnlStorageTraits< false >,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{};

/****
 * Mesh initializer layer for vertices
 * - vertices must always be stored
 * - their orientation does not make sense
 */
template< typename MeshConfig >
class tnlMeshInitializerLayer< MeshConfig,
                               tnlDimensionsTag< 0 >,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
{
   typedef tnlMesh< MeshConfig >                                        MeshType;
   typedef tnlDimensionsTag< 0 >                                    DimensionsTag;

   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >        Tag;
   typedef typename Tag::Tag                                           EntityTag;
   typedef typename Tag::StorageArrayType                              ContainerType;
   typedef typename Tag::AccessArrayType                               SharedContainerType;
   typedef typename ContainerType::IndexType                           GlobalIndexType;

   typedef typename tnlMeshTraits< MeshConfig >::CellTopology           CellTopology;

   typedef tnlMeshInitializer< MeshConfig >                             InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, 
                                     typename MeshConfig::CellTopology >     CellInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >            VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType > VertexInitializerContainerType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType    LocalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType           PointArrayType;
   typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag::value >            EntityTraits;
   typedef typename EntityTraits::StorageArrayType                          EntityArrayType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTag >                EntityInitializerType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTag >  SuperentityInitializerType;

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
      
      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds ){};
      
      void initEntities( InitializerType& initializer, const PointArrayType& points )
      {
         EntityArrayType &vertexArray = initializer.template meshEntitiesArray< DimensionsTag >();
         vertexArray.setSize( points.getSize() );
         for( GlobalIndexType i = 0; i < vertexArray.getSize(); i++ )
            EntityInitializerType::setVertexPoint( vertexArray[i], points[i], initializer );

         superentityInitializer.initSuperentities( initializer );
      }
         
      void findEntitySeedIndex() const                               {} // This method is due to 'using BaseType::findEntityIndex;' in the derived class.

      void createEntityInitializers()
      {
         vertexInitializerContainer.setSize( this->getMesh().template getNumberOfEntities< DimensionsTag::value >() );
      }
      
      SuperentityInitializerType& getSuperentityInitializer( DimensionsTag )
      {
         return this->superentityInitializer;
      }

      void createEntityReferenceOrientations() const {}
      
      void getReferenceOrientation() const {}
      
   private:
      
      SuperentityInitializerType superentityInitializer;

      VertexInitializerContainerType vertexInitializerContainer;

      MeshType* mesh;
};




#endif /* TNLMESHINITIALIZER_H_ */
