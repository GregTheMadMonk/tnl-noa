/***************************************************************************
                          tnlMeshInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlMeshEntityTraits.h>
#include <mesh/traits/tnlMeshSubentityTraits.h>
#include <mesh/traits/tnlMeshSuperentityTraits.h>
#include <mesh/initializer/tnlMeshEntityInitializer.h>
#include <mesh/tnlMesh.h>
#include <mesh/initializer/tnlMeshSubentitySeedCreator.h>
#include <mesh/initializer/tnlMeshSuperentityStorageInitializer.h>
#include <mesh/tnlMeshEntityReferenceOrientation.h>
#include <mesh/initializer/tnlMeshEntitySeed.h>
#include <mesh/initializer/tnlMeshEntitySeedKey.h>

namespace TNL {

template< typename MeshConfig >
class tnlMesh;

template< typename MeshConfig,
          typename DimensionsTag,
          bool EntityStorage =
             tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::storageEnabled,
          bool EntityReferenceOrientationStorage =
             tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::orientationNeeded >
class tnlMeshInitializerLayer;


template< typename MeshConfig,
          typename EntityTopology>
class tnlMeshEntityInitializer;

template< typename MeshConfig >
class tnlMeshInitializer
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename tnlMeshTraits< MeshConfig >::DimensionsTag >
{
   public:
 
      typedef tnlMesh< MeshConfig >                                  MeshType;
      typedef tnlMeshTraits< MeshConfig >                            MeshTraits;
      static const int Dimensions = MeshTraits::meshDimensions;
      typedef tnlDimensionsTag< Dimensions >                         DimensionsTag;
      typedef tnlMeshInitializerLayer< MeshConfig, DimensionsTag >   BaseType;
      typedef typename MeshTraits::PointArrayType                    PointArrayType;
      typedef typename MeshTraits::CellSeedArrayType                 CellSeedArrayType;
      typedef typename MeshTraits::GlobalIndexType                   GlobalIndexType;
 
      template< typename DimensionsTag, typename SuperdimensionsTag > using SuperentityStorageNetwork =
      typename MeshTraits::template SuperentityTraits<
         typename MeshTraits::template EntityTraits< DimensionsTag::value >::EntityTopology,
         SuperdimensionsTag::value >::StorageNetworkType;


      tnlMeshInitializer()
      : verbose( false ), mesh( 0 )
      {}

      void setVerbose( bool verbose )
      {
         this->verbose = verbose;
      }

      bool createMesh( const PointArrayType& points,
                       const CellSeedArrayType& cellSeeds,
                       MeshType& mesh )
      {
        std::cout << "======= Starting mesh initiation ========" << std::endl;
         this->mesh = &mesh;

        std::cout << "========= Creating entity seeds =============" << std::endl;
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );

        std::cout << "========= Creating entity reference orientations =============" << std::endl;
         BaseType::createEntityReferenceOrientations();

        std::cout << "====== Initiating entities ==============" << std::endl;
         BaseType::initEntities( *this, points, cellSeeds );

         return true;
      }

      template<typename SubDimensionsTag, typename EntityType >
      static typename MeshTraits::template SubentityTraits< typename EntityType::EntityTopology, SubDimensionsTag::value >::IdArrayType&
      subentityIdsArray( EntityType& entity )
      {
         return entity.template subentityIdsArray< SubDimensionsTag::value >();
      }

      template< typename SuperDimensionsTag, typename MeshEntity>
      static typename MeshTraits::IdArrayAccessorType&
      superentityIdsArray( MeshEntity& entity )
      {
         return entity.template superentityIdsArray< SuperDimensionsTag::value >();
      }

      template<typename SubDimensionsTag, typename MeshEntity >
      static typename MeshTraits::template SubentityTraits< typename MeshEntity::EntityTopology, SubDimensionsTag::value >::OrientationArrayType&
      subentityOrientationsArray( MeshEntity &entity )
      {
         return entity.template subentityOrientationsArray< SubDimensionsTag::value >();
      }

      template< typename DimensionsTag >
      typename MeshTraits::template EntityTraits< DimensionsTag::value >::StorageArrayType&
      meshEntitiesArray()
      {
         return mesh->template entitiesArray< DimensionsTag >();
      }

      template< typename DimensionsTag, typename SuperDimensionsTag >
      typename MeshTraits::GlobalIdArrayType&
      meshSuperentityIdsArray()
      {
         return mesh->template superentityIdsArray< DimensionsTag, SuperDimensionsTag >();
      }
 
      template< typename EntityTopology, typename SuperdimensionsTag >
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
      meshSuperentityStorageNetwork()
      {
         return mesh->template getSuperentityStorageNetwork< EntityTopology, SuperdimensionsTag >();
      }

      static void
      setVertexPoint( typename MeshType::VertexType& vertex, const typename MeshType::PointType& point )
      {
         vertex.setPoint( point );
      }

      template< typename DimensionsTag >
      tnlMeshSuperentityStorageInitializer< MeshConfig, typename MeshTraits::template EntityTraits< DimensionsTag::value >::EntityTopology >&
      getSuperentityInitializer()
      {
         return BaseType::getSuperentityInitializer( DimensionsTag() );
      }

 
      template< typename DimensionsTag >
      const tnlMeshEntityReferenceOrientation< MeshConfig, typename MeshTraits::template EntityTraits< DimensionsTag::value >::EntityTopology >&
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
                               true,
                               false >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename tnlMeshTraits< MeshConfig >::DimensionsTag::Decrement >
{
   typedef tnlMeshTraits< MeshConfig >                                              MeshTraits;
   static const int Dimensions = MeshTraits::meshDimensions;
   typedef tnlDimensionsTag< Dimensions >                                           DimensionsTag;
   typedef tnlMeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement > BaseType;

   typedef tnlMesh< MeshConfig >                                                    MeshType;
   typedef typename MeshTraits::template EntityTraits< Dimensions >                 EntityTraits;
   typedef typename EntityTraits::EntityTopology                                    EntityTopology;
   typedef typename MeshTraits::GlobalIndexType                                     GlobalIndexType;
   typedef typename MeshTraits::CellTopology                                        CellTopology;
   typedef typename EntityTraits::StorageArrayType                                  StorageArrayType;

   typedef tnlMeshInitializer< MeshConfig >                                         InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                   EntityInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                   CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >                CellInitializerContainerType;
   typedef typename MeshTraits::CellSeedArrayType                                   CellSeedArrayType;
   typedef typename MeshTraits::LocalIndexType                                      LocalIndexType;
   typedef typename MeshTraits::PointArrayType                                      PointArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, CellTopology >                            SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTopology >      SuperentityInitializerType;

   public:

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      void initEntities( InitializerType &initializer, const PointArrayType &points, const CellSeedArrayType &cellSeeds)
      {
         StorageArrayType &entityArray = initializer.template meshEntitiesArray< DimensionsTag >();
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         entityArray.setSize( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < entityArray.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << std::endl;
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
         typedef typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentitiesTraits< 0 >::LocalIndexType LocalIndexType;
         const GlobalIndexType numberOfVertices( this->getMesh().getNumberOfVertices() );
         for( GlobalIndexType cell = 0;
              cell < this->getMesh().template getEntitiesCount< typename MeshType::Cell >();
              cell++ )
            for( LocalIndexType i = 0;
                 i < this->getMesh().getCell( cell ).getNumberOfVertices();
                 i++ )
            {
               if( this->getMesh().getCell( cell ).getVerticesIndices()[ i ] == - 1 )
               {
                  std::cerr << "The cell number " << cell << " does not have properly set vertex index number " << i << "." << std::endl;
                  return false;
               }
               if( this->getMesh().getCell( cell ).getVerticesIndices()[ i ] >= numberOfVertices )
               {
                  std::cerr << "The cell number " << cell << " does not have properly set vertex index number " << i
                       << ". The index " << this->getMesh().getCell( cell ).getVerticesIndices()[ i ]
                       << "is higher than the number of all vertices ( " << numberOfVertices
                       << " )." << std::endl;
                  return false;
               }
            }
         return true;
      }

   private:
      typedef  typename tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;

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
                               true,
                               false >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
      typedef tnlMeshTraits< MeshConfig >                                              MeshTraits;
   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement > BaseType;

   typedef tnlMesh< MeshConfig >                                                    MeshType;
   typedef typename MeshTraits::template EntityTraits< Dimensions >                 EntityTraits;
   typedef typename EntityTraits::EntityTopology                                    EntityTopology;
   typedef typename MeshTraits::GlobalIndexType                                     GlobalIndexType;
   typedef typename MeshTraits::CellTopology                                        CellTopology;
   typedef typename EntityTraits::StorageArrayType                                  StorageArrayType;

   typedef tnlMeshInitializer< MeshConfig >                                         InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                   EntityInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >                   CellInitializerType;
   typedef tnlArray< CellInitializerType, tnlHost, GlobalIndexType >                CellInitializerContainerType;
   typedef typename EntityTraits::SeedArrayType                                     EntitySeedArrayType;
   typedef typename MeshTraits::CellSeedArrayType                                   CellSeedArrayType;
   typedef typename MeshTraits::LocalIndexType                                      LocalIndexType;
   typedef typename MeshTraits::PointArrayType                                      PointArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                          SeedType;
   typedef  tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTopology >      SuperentityInitializerType;

   typedef typename
      tnlMeshSubentityTraits< MeshConfig,
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
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << std::endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << std::endl;
               //tnlMeshEntitySeed< tnlMeshConfigBase< CellTopology >, EntityTopology >& entitySeed = subentytiSeeds[ j ];
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
         StorageArrayType &entityArray = initializer.template meshEntitiesArray< DimensionsTag >();
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         entityArray.setSize( this->seedsIndexedSet.getSize() );
         EntitySeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << std::endl;
            EntityInitializerType::initEntity( entityArray[ i ], i, seedsArray[ i ], initializer );
         }
         this->seedsIndexedSet.reset();

         this->superentityInitializer.initSuperentities( initializer );

         BaseType::initEntities(initializer, points);
      }

      void createEntityReferenceOrientations() const {}
   private:
 
      typedef  typename tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
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
                               true,
                               true >
   : public tnlMeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
   typedef tnlMeshInitializerLayer< MeshConfig,
                                    typename DimensionsTag::Decrement >       BaseType;
   typedef tnlMesh< MeshConfig >                                              MeshType;
   typedef typename MeshType::MeshTraits                                      MeshTraits;

   typedef typename MeshType::template EntityTraits< DimensionsTag::value >   EntityTraits;
   typedef typename EntityTraits::EntityTopology                              EntityTopology;
   typedef typename EntityTraits::EntityType                                  EntityType;
   typedef typename EntityTraits::StorageArrayType                            ContainerType;
   typedef typename EntityTraits::UniqueContainerType                         UniqueContainerType;
   typedef typename ContainerType::IndexType                                  GlobalIndexType;
   typedef typename MeshTraits::CellTopology                                  CellTopology;

   typedef tnlMeshInitializer< MeshConfig >                                   InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, CellTopology >               CellInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >             EntityInitializerType;
   typedef tnlArray< EntityInitializerType, tnlHost, GlobalIndexType >        EntityInitializerContainerType;
   typedef typename MeshTraits::CellSeedArrayType                             CellSeedArrayType;
   typedef typename MeshTraits::LocalIndexType                                LocalIndexType;
   typedef typename MeshTraits::PointArrayType                                PointArrayType;
   typedef typename EntityTraits::StorageArrayType                            EntityArrayType;
   typedef typename EntityTraits::SeedArrayType                               SeedArrayType;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                    SeedType;
   typedef tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTopology > SuperentityInitializerType;
   typedef typename EntityTraits::ReferenceOrientationType                    ReferenceOrientationType;
   typedef typename EntityTraits::ReferenceOrientationArrayType               ReferenceOrientationArrayType;


   typedef typename
      tnlMeshSubentityTraits< MeshConfig,
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
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << std::endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << std::endl;
               //tnlMeshEntitySeed< tnlMeshConfigBase< CellTopology >, EntityTopology >& entitySeed = subentytiSeeds[ j ];
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
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         entityArray.setSize( this->seedsIndexedSet.getSize() );
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //cout << "  Initiating entity " << i << std::endl;
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
         //cout << " Creating entity reference orientations with " << DimensionsTag::value << " dimensions ... " << std::endl;
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         this->referenceOrientations.setSize( seedsArray.getSize() );
         for( GlobalIndexType i = 0; i < seedsArray.getSize(); i++ )
         {
            //cout << "  Creating reference orientation for entity " << i << std::endl;
            this->referenceOrientations[ i ] = ReferenceOrientationType( seedsArray[ i ] );
         }
         BaseType::createEntityReferenceOrientations();
		}	
 
   private:
 
      typedef  typename tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
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
                               false,
                               false >
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
                               true,
                               false >
{
   typedef tnlMesh< MeshConfig >                                              MeshType;
   typedef typename MeshType::MeshTraits                                      MeshTraits;
   typedef tnlDimensionsTag< 0 >                                              DimensionsTag;

   typedef typename MeshType::template EntityTraits< DimensionsTag::value >   EntityTraits;
   typedef typename EntityTraits::EntityTopology                              EntityTopology;
   typedef typename EntityTraits::StorageArrayType                            ContainerType;
   typedef typename EntityTraits::AccessArrayType                             SharedContainerType;
   typedef typename ContainerType::IndexType                                  GlobalIndexType;

   typedef typename MeshTraits::CellTopology                                  CellTopology;

   typedef tnlMeshInitializer< MeshConfig >                                   InitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, CellTopology >               CellInitializerType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >             VertexInitializerType;
   typedef tnlArray< VertexInitializerType, tnlHost, GlobalIndexType >        VertexInitializerContainerType;
   typedef typename tnlMeshTraits< MeshConfig >::CellSeedArrayType            CellSeedArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType               LocalIndexType;
   typedef typename tnlMeshTraits< MeshConfig >::PointArrayType               PointArrayType;
   typedef typename EntityTraits::StorageArrayType                            EntityArrayType;
   typedef tnlMeshEntityInitializer< MeshConfig, EntityTopology >             EntityInitializerType;
   typedef tnlMeshSuperentityStorageInitializer< MeshConfig, EntityTopology > SuperentityInitializerType;

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
                  std::cerr << " index = " << index
                       << " vertexInitializerContainer.getSize() = " << vertexInitializerContainer.getSize() << std::endl; );
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

} // namespace TNL
