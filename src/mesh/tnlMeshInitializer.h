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

template< typename ConfigTag,
          typename DimensionsTag,
          typename EntityStorageTag = 
             typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::EntityStorageTag,
          typename EntityReferenceOrientationStorage = 
             tnlStorageTraits< tnlMeshConfigTraits< ConfigTag >::template EntityTraits< DimensionsTag >::orientationNeeded > >
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
		
      cout << "========= Creating entity reference orientations =============" << endl;
      BaseType::createEntityReferenceOrientations();
      
      cout << "====== Initiating entities ==============" << endl;
      BaseType::initEntities( *this, points, cellSeeds );
      
      return true;
   }
   
   template<typename SubDimensionsTag, typename EntityType >
   static typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< typename EntityType::Tag, SubDimensionsTag >::IdArrayType&
   subentityIdsArray( EntityType& entity )
   {
      return entity.template subentityIdsArray< SubDimensionsTag >();
   }
   
   template< typename SuperDimensionsTag, typename MeshEntity>
   static typename tnlMeshConfigTraits< ConfigTag >::IdArrayAccessorType&
   superentityIdsArray( MeshEntity& entity )
   {
      return entity.template superentityIdsArray< SuperDimensionsTag >();
   }
   
   template<typename SubDimensionsTag, typename MeshEntity >
	static typename tnlMeshConfigTraits< ConfigTag >::template SubentityTraits< typename MeshEntity::Tag, SubDimensionsTag >::OrientationArrayType&
   subentityOrientationsArray( MeshEntity &entity )
   {
      return entity.template subentityOrientationsArray< SubDimensionsTag >();
   }
   
   template< typename DimensionsTag >
   typename tnlMeshConfigTraits< ConfigTag >::template EntityTraits< DimensionsTag>::ContainerType&
   meshEntitiesArray()
   {
      return mesh->template entitiesArray< DimensionsTag >();
   }
   
   template< typename DimensionsTag, typename SuperDimensionsTag >
	typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType&
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
   tnlMeshSuperentityStorageInitializer< ConfigTag, typename tnlMeshConfigTraits< ConfigTag >::template EntityTraits< DimensionsTag >::Tag >&
   getSuperentityInitializer()
   {
      return BaseType::getSuperentityInitializer( DimensionsTag() );
   }

   typedef typename tnlMeshTraits< ConfigTag >::GlobalIndexType GlobalIndexType;
   template< typename DimensionsTag >
	const tnlMeshEntityReferenceOrientation< ConfigTag, typename tnlMeshConfigTraits< ConfigTag >::template EntityTraits< DimensionsTag >::Tag >&
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
template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               typename tnlMeshTraits< ConfigTag >::DimensionsTag,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename tnlMeshTraits< ConfigTag >::DimensionsTag::Decrement >
{
   typedef typename tnlMeshTraits< ConfigTag >::DimensionsTag        DimensionsTag;

   typedef tnlMeshInitializerLayer< ConfigTag,
                                    typename DimensionsTag::Decrement >   BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >            EntityTraits;
   typedef typename EntityTraits::Tag                                            EntityTag;
   typedef typename EntityTraits::ContainerType                                  ContainerType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellTopology      CellTopology;
   typedef typename EntityTraits::ContainerType                          EntityArrayType;

   typedef tnlMeshInitializer< ConfigTag >                              InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >             EntityInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >             CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >    CellInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;
   typedef typename tnlMeshTraits< ConfigTag >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitySeed< ConfigTag, CellTopology >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< ConfigTag, EntityTag >  SuperentityInitializerType;

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

   private:
      typedef  typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;      

      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned off
 */
template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTag,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
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
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellTopology          CellTopology;

   typedef tnlMeshInitializer< ConfigTag >                                 InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     typename ConfigTag::CellTopology >         CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType    CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType       LocalIndexType;
   typedef typename tnlMeshTraits< ConfigTag >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >            EntityTraits;
   typedef typename EntityTraits::ContainerType                          EntityArrayType;
   typedef typename EntityTraits::SeedArrayType                          SeedArrayType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< ConfigTag, EntityTag >  SuperentityInitializerType;


   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                typename ConfigTag::CellTopology,
                                DimensionsTag >::SubentityContainerType SubentitiesContainerType;

   public:

      using BaseType::getEntityInitializer;
      EntityInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         //return entityInitializerContainer[ index ];
      }

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef tnlMeshSubentitySeedsCreator< ConfigTag, CellTopology, DimensionsTag >  SubentitySeedsCreator;
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
      
      typedef  typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned on
 */
template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTag,
                               tnlStorageTraits< true >,
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
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellTopology          CellTopology;

   typedef tnlMeshInitializer< ConfigTag >                                 InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag,
                                     typename ConfigTag::CellTopology >         CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >     EntityInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType    CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType       LocalIndexType;
   typedef typename tnlMeshTraits< ConfigTag >::PointArrayType          PointArrayType;
   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >            EntityTraits;
   typedef typename EntityTraits::ContainerType                          EntityArrayType;
   typedef typename EntityTraits::SeedArrayType                          SeedArrayType;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >                 SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< ConfigTag, EntityTag >  SuperentityInitializerType;
   typedef typename EntityTraits::ReferenceOrientationType               ReferenceOrientationType;
   typedef typename EntityTraits::ReferenceOrientationArrayType          ReferenceOrientationArrayType;


   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                typename ConfigTag::CellTopology,
                                DimensionsTag >::SubentityContainerType SubentitiesContainerType;

   public:      
      
      using BaseType::getEntityInitializer;
      EntityInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         //return entityInitializerContainer[ index ];
      }

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef tnlMeshSubentitySeedsCreator< ConfigTag, CellTopology, DimensionsTag >  SubentitySeedsCreator;
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
      
      typedef  typename tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
      ReferenceOrientationArrayType referenceOrientations;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshInitializerLayer< ConfigTag,
                               DimensionsTag,
                               tnlStorageTraits< false >,
                               tnlStorageTraits< false > >
   : public tnlMeshInitializerLayer< ConfigTag,
                                     typename DimensionsTag::Decrement >
{};

/****
 * Mesh initializer layer for vertices
 * - vertices must always be stored
 * - their orientation does not make sense
 */
template< typename ConfigTag >
class tnlMeshInitializerLayer< ConfigTag,
                               tnlDimensionsTag< 0 >,
                               tnlStorageTraits< true >,
                               tnlStorageTraits< false > >
{
   typedef tnlMesh< ConfigTag >                                        MeshType;
   typedef tnlDimensionsTag< 0 >                                    DimensionsTag;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >        Tag;
   typedef typename Tag::Tag                                           EntityTag;
   typedef typename Tag::ContainerType                                 ContainerType;
   typedef typename Tag::SharedContainerType                           SharedContainerType;
   typedef typename ContainerType::IndexType                           GlobalIndexType;

   typedef typename tnlMeshTraits< ConfigTag >::CellTopology           CellTopology;

   typedef tnlMeshInitializer< ConfigTag >                             InitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, 
                                     typename ConfigTag::CellTopology >     CellInitializerType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >            VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType > VertexInitializerContainerType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::CellSeedArrayType CellSeedArrayType;
   typedef typename tnlMeshConfigTraits< ConfigTag >::LocalIndexType    LocalIndexType;
   typedef typename tnlMeshTraits< ConfigTag >::PointArrayType           PointArrayType;
   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTag >            EntityTraits;
   typedef typename EntityTraits::ContainerType                          EntityArrayType;
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                EntityInitializerType;
   typedef  tnlMeshSuperentityStorageInitializer< ConfigTag, EntityTag >  SuperentityInitializerType;

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
