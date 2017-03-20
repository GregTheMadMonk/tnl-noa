/***************************************************************************
                          MeshInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDimensionsTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntityInitializer.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshSubentitySeedCreator.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshSuperentityStorageInitializer.h>
#include <TNL/Meshes/MeshDetails/MeshEntityReferenceOrientation.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeed.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeedKey.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class Mesh;

template< typename MeshConfig,
          typename DimensionsTag,
          bool EntityStorage =
             MeshEntityTraits< MeshConfig, DimensionsTag::value >::storageEnabled,
          bool EntityReferenceOrientationStorage =
             MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::orientationNeeded >
class MeshInitializerLayer;


template< typename MeshConfig,
          typename EntityTopology>
class MeshEntityInitializer;

template< typename MeshConfig >
class MeshInitializer
   : public MeshInitializerLayer< MeshConfig,
                                     typename MeshTraits< MeshConfig >::DimensionsTag >
{
   public:
 
      typedef Mesh< MeshConfig >                                  MeshType;
      typedef MeshTraits< MeshConfig >                            MeshTraitsType;
      static const int Dimensions = MeshTraitsType::meshDimensions;
      typedef MeshDimensionsTag< Dimensions >                      DimensionsTag;
      typedef MeshInitializerLayer< MeshConfig, DimensionsTag >   BaseType;
      typedef typename MeshTraitsType::PointArrayType             PointArrayType;
      typedef typename MeshTraitsType::CellSeedArrayType          CellSeedArrayType;
      typedef typename MeshTraitsType::GlobalIndexType            GlobalIndexType;
 
      template< typename DimensionsTag, typename SuperdimensionsTag > using SuperentityStorageNetwork =
      typename MeshTraitsType::template SuperentityTraits<
         typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::EntityTopology,
         SuperdimensionsTag::value >::StorageNetworkType;


      MeshInitializer()
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
      static typename MeshTraitsType::template SubentityTraits< typename EntityType::EntityTopology, SubDimensionsTag::value >::IdArrayType&
      subentityIdsArray( EntityType& entity )
      {
         return entity.template subentityIdsArray< SubDimensionsTag::value >();
      }

      template< typename SuperDimensionsTag, typename MeshEntity>
      static typename MeshTraitsType::IdArrayAccessorType&
      superentityIdsArray( MeshEntity& entity )
      {
         return entity.template superentityIdsArray< SuperDimensionsTag::value >();
      }

      template<typename SubDimensionsTag, typename MeshEntity >
      static typename MeshTraitsType::template SubentityTraits< typename MeshEntity::EntityTopology, SubDimensionsTag::value >::OrientationArrayType&
      subentityOrientationsArray( MeshEntity &entity )
      {
         return entity.template subentityOrientationsArray< SubDimensionsTag::value >();
      }

      template< typename DimensionsTag >
      typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::StorageArrayType&
      meshEntitiesArray()
      {
         return mesh->template entitiesArray< DimensionsTag >();
      }

      template< typename DimensionsTag, typename SuperDimensionsTag >
      typename MeshTraitsType::GlobalIdArrayType&
      meshSuperentityIdsArray()
      {
         return mesh->template superentityIdsArray< DimensionsTag, SuperDimensionsTag >();
      }
 
      template< typename EntityTopology, typename SuperdimensionsTag >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
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
      MeshSuperentityStorageInitializer< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::EntityTopology >&
      getSuperentityInitializer()
      {
         return BaseType::getSuperentityInitializer( DimensionsTag() );
      }

 
      template< typename DimensionsTag >
      const MeshEntityReferenceOrientation< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::EntityTopology >&
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
class MeshInitializerLayer< MeshConfig,
                               typename MeshTraits< MeshConfig >::DimensionsTag,
                               true,
                               false >
   : public MeshInitializerLayer< MeshConfig,
                                     typename MeshTraits< MeshConfig >::DimensionsTag::Decrement >
{
   typedef MeshTraits< MeshConfig >                                              MeshTraitsType;
   static const int Dimensions = MeshTraitsType::meshDimensions;
   typedef MeshDimensionsTag< Dimensions >                                        DimensionsTag;
   typedef MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement > BaseType;

   typedef Mesh< MeshConfig >                                                    MeshType;
   typedef typename MeshTraitsType::template EntityTraits< Dimensions >          EntityTraitsType;
   typedef typename EntityTraitsType::EntityTopology                             EntityTopology;
   typedef typename MeshTraitsType::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshTraitsType::CellTopology                                 CellTopology;
   typedef typename EntityTraitsType::StorageArrayType                           StorageArrayType;

   typedef MeshInitializer< MeshConfig >                                         InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                   EntityInitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                   CellInitializerType;
   typedef Containers::Array< CellInitializerType, Devices::Host, GlobalIndexType >  CellInitializerContainerType;
   typedef typename MeshTraitsType::CellSeedArrayType                            CellSeedArrayType;
   typedef typename MeshTraitsType::LocalIndexType                               LocalIndexType;
   typedef typename MeshTraitsType::PointArrayType                               PointArrayType;
   typedef MeshEntitySeed< MeshConfig, CellTopology >                            SeedType;
   typedef  MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >      SuperentityInitializerType;

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
         typedef typename MeshEntity< MeshConfig, EntityTopology >::template SubentitiesTraits< 0 >::LocalIndexType LocalIndexType;
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
      typedef  typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;

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
class MeshInitializerLayer< MeshConfig,
                               DimensionsTag,
                               true,
                               false >
   : public MeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
      typedef MeshTraits< MeshConfig >                                           MeshTraitsType;
   static const int Dimensions = DimensionsTag::value;
   typedef MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement > BaseType;

   typedef Mesh< MeshConfig >                                                    MeshType;
   typedef typename MeshTraitsType::template EntityTraits< Dimensions >          EntityTraitsType;
   typedef typename EntityTraitsType::EntityTopology                             EntityTopology;
   typedef typename MeshTraitsType::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshTraitsType::CellTopology                                 CellTopology;
   typedef typename EntityTraitsType::StorageArrayType                           StorageArrayType;

   typedef MeshInitializer< MeshConfig >                                         InitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                   EntityInitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                   CellInitializerType;
   typedef Containers::Array< CellInitializerType, Devices::Host, GlobalIndexType >  CellInitializerContainerType;
   typedef typename EntityTraitsType::SeedArrayType                              EntitySeedArrayType;
   typedef typename MeshTraitsType::CellSeedArrayType                            CellSeedArrayType;
   typedef typename MeshTraitsType::LocalIndexType                               LocalIndexType;
   typedef typename MeshTraitsType::PointArrayType                               PointArrayType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                          SeedType;
   typedef MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >       SuperentityInitializerType;

   typedef typename
      MeshSubentityTraits< MeshConfig,
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
         typedef MeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionsTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << std::endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << std::endl;
               //MeshEntitySeed< MeshConfigBase< CellTopology >, EntityTopology >& entitySeed = subentytiSeeds[ j ];
               this->seedsIndexedSet.insert( subentytiSeeds[ j ] );
            }
         }
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed ) const
      {
         // FIXME: index may be uninitialized (when seedsIndexedSet.find returns false)
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
 
      typedef  typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
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
class MeshInitializerLayer< MeshConfig,
                            DimensionsTag,
                            true,
                            true >
   : public MeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{
   typedef MeshInitializerLayer< MeshConfig,
                                    typename DimensionsTag::Decrement >       BaseType;
   typedef Mesh< MeshConfig >                                                 MeshType;
   typedef typename MeshType::MeshTraitsType                                  MeshTraitsType;

   typedef typename MeshType::template EntityTraits< DimensionsTag::value >   EntityTraitsType;
   typedef typename EntityTraitsType::EntityTopology                          EntityTopology;
   typedef typename EntityTraitsType::EntityType                              EntityType;
   typedef typename EntityTraitsType::StorageArrayType                        ContainerType;
   typedef typename EntityTraitsType::UniqueContainerType                     UniqueContainerType;
   typedef typename ContainerType::IndexType                                  GlobalIndexType;
   typedef typename MeshTraitsType::CellTopology                              CellTopology;

   typedef MeshInitializer< MeshConfig >                                      InitializerType;
   typedef MeshEntityInitializer< MeshConfig, CellTopology >                  CellInitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                EntityInitializerType;
   typedef Containers::Array< EntityInitializerType, Devices::Host, GlobalIndexType >        EntityInitializerContainerType;
   typedef typename MeshTraitsType::CellSeedArrayType                         CellSeedArrayType;
   typedef typename MeshTraitsType::LocalIndexType                            LocalIndexType;
   typedef typename MeshTraitsType::PointArrayType                            PointArrayType;
   typedef typename EntityTraitsType::StorageArrayType                        EntityArrayType;
   typedef typename EntityTraitsType::SeedArrayType                           SeedArrayType;
   typedef MeshEntitySeed< MeshConfig, EntityTopology >                       SeedType;
   typedef MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >    SuperentityInitializerType;
   typedef typename EntityTraitsType::ReferenceOrientationType                ReferenceOrientationType;
   typedef typename EntityTraitsType::ReferenceOrientationArrayType           ReferenceOrientationArrayType;


   typedef typename
      MeshSubentityTraits< MeshConfig,
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
         typedef MeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionsTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << std::endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << std::endl;
               //MeshEntitySeed< MeshConfigBase< CellTopology >, EntityTopology >& entitySeed = subentytiSeeds[ j ];
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
 
      typedef  typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType                     SeedIndexedSet;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
      ReferenceOrientationArrayType referenceOrientations;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename MeshConfig,
          typename DimensionsTag >
class MeshInitializerLayer< MeshConfig,
                               DimensionsTag,
                               false,
                               false >
   : public MeshInitializerLayer< MeshConfig,
                                     typename DimensionsTag::Decrement >
{};

/****
 * Mesh initializer layer for vertices
 * - vertices must always be stored
 * - their orientation does not make sense
 */
template< typename MeshConfig >
class MeshInitializerLayer< MeshConfig,
                               MeshDimensionsTag< 0 >,
                               true,
                               false >
{
   typedef Mesh< MeshConfig >                                                 MeshType;
   typedef typename MeshType::MeshTraitsType                                  MeshTraitsType;
   typedef MeshDimensionsTag< 0 >                                              DimensionsTag;

   typedef typename MeshType::template EntityTraits< DimensionsTag::value >   EntityTraitsType;
   typedef typename EntityTraitsType::EntityTopology                          EntityTopology;
   typedef typename EntityTraitsType::StorageArrayType                        ContainerType;
   typedef typename EntityTraitsType::AccessArrayType                         SharedContainerType;
   typedef typename ContainerType::IndexType                                  GlobalIndexType;

   typedef typename MeshTraitsType::CellTopology                                       CellTopology;

   typedef MeshInitializer< MeshConfig >                                           InitializerType;
   typedef MeshEntityInitializer< MeshConfig, CellTopology >                       CellInitializerType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >                     VertexInitializerType;
   typedef Containers::Array< VertexInitializerType, Devices::Host, GlobalIndexType >  VertexInitializerContainerType;
   typedef typename MeshTraits< MeshConfig >::CellSeedArrayType            CellSeedArrayType;
   typedef typename MeshTraits< MeshConfig >::LocalIndexType               LocalIndexType;
   typedef typename MeshTraits< MeshConfig >::PointArrayType               PointArrayType;
   typedef typename EntityTraitsType::StorageArrayType                            EntityArrayType;
   typedef MeshEntityInitializer< MeshConfig, EntityTopology >             EntityInitializerType;
   typedef MeshSuperentityStorageInitializer< MeshConfig, EntityTopology > SuperentityInitializerType;

   public:

      void setMesh( MeshType& mesh )
      {
         this->mesh = &mesh;
      }

      MeshType& getMesh()
      {
         TNL_ASSERT( this->mesh, );
         return *( this->mesh );
      }

      VertexInitializerType& getEntityInitializer( DimensionsTag, GlobalIndexType index )
      {
         TNL_ASSERT( index >= 0 && index < vertexInitializerContainer.getSize(),
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

} // namespace Meshes
} // namespace TNL
