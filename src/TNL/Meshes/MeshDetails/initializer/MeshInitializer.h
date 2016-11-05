/***************************************************************************
                          MeshInitializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Meshes/MeshDimensionTag.h>
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
          typename DimensionTag,
          bool EntityStorage =
             MeshEntityTraits< MeshConfig, DimensionTag::value >::storageEnabled,
          bool EntityReferenceOrientationStorage =
             // FIXME
//             MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::orientationNeeded >
             false >
class MeshInitializerLayer;


template< typename MeshConfig,
          typename EntityTopology>
class MeshEntityInitializer;

template< typename MeshConfig >
class MeshInitializer
   : public MeshInitializerLayer< MeshConfig,
                                  typename MeshTraits< MeshConfig >::DimensionsTag >
{
   protected:
      // must be declared before its use in expression with decltype()
      Mesh< MeshConfig >* mesh;

   public:
      using MeshType          = Mesh< MeshConfig >;
      using MeshTraitsType    = MeshTraits< MeshConfig >;
      using DimensionsTag     = MeshDimensionsTag< MeshTraitsType::meshDimensions >;
      using BaseType          = MeshInitializerLayer< MeshConfig, DimensionsTag >;
      using PointArrayType    = typename MeshTraitsType::PointArrayType;
      using CellSeedArrayType = typename MeshTraitsType::CellSeedArrayType;
      using GlobalIndexType   = typename MeshTraitsType::GlobalIndexType;


      MeshInitializer()
      : mesh( 0 )
      {}

      // The points and cellSeeds arrays will be reset when not needed to save memory.
      bool createMesh( PointArrayType& points,
                       CellSeedArrayType& cellSeeds,
                       MeshType& mesh )
      {
         this->mesh = &mesh;
         BaseType::initEntities( *this, points, cellSeeds, mesh );
         return true;
      }

      template< typename Entity, typename GlobalIndex >
      void setEntityId( Entity& entity, const GlobalIndex& index )
      {
         entity.setId( index );
      }

      template< int Dimensions >
      bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
      {
         //std::cout << "Setting number of entities with " << Dimensions << " dimension to " << entitiesCount << std::endl;
         return mesh->template setNumberOfEntities< Dimensions >( entitiesCount );
      }

      template< int Subdimensions, typename EntityType, typename LocalIndex, typename GlobalIndex >
      void
      setSubentityIndex( const EntityType& entity, const GlobalIndex& entityIndex, const LocalIndex& localIndex, const GlobalIndex& globalIndex )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimensions, Subdimensions >().getValues( entityIndex )[ localIndex ] = globalIndex;
      }

      template< int Subdimensions, typename EntityType, typename LocalIndex, typename GlobalIndex >
      GlobalIndex
      getSubentityIndex( const EntityType& entity, const GlobalIndex& entityIndex, const LocalIndex& localIndex )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         return mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimensions, Subdimensions >().getValues( entityIndex )[ localIndex ];
      }

      template< typename EntityType, typename GlobalIndex >
      auto
      getSubvertices( const EntityType& entity, const GlobalIndex& entityIndex )
         -> decltype( this->mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimensions, 0 >().getValues( 0 ) )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         return mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimensions, 0 >().getValues( entityIndex );
      }

      template<typename SubDimensionsTag, typename MeshEntity >
      static typename MeshTraitsType::template SubentityTraits< typename MeshEntity::EntityTopology, SubDimensionsTag::value >::OrientationArrayType&
      subentityOrientationsArray( MeshEntity& entity )
      {
         return entity.template subentityOrientationsArray< SubDimensionTag::value >();
      }

      template< typename EntityTopology, int Superdimensions >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimensions >::StorageNetworkType&
      meshSuperentityStorageNetwork()
      {
         return mesh->template getSuperentityStorageNetwork< EntityTopology::dimensions, Superdimensions >();
      }

      static void
      setVertexPoint( typename MeshType::VertexType& vertex, const typename MeshType::PointType& point )
      {
         vertex.setPoint( point );
      }


      template< typename DimensionsTag >
      const MeshEntityReferenceOrientation< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::EntityTopology >&
      getReferenceOrientation( GlobalIndexType index ) const
      {
         return BaseType::getReferenceOrientation( DimensionsTag(), index );
      }
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
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using DimensionsTag         = typename MeshTraitsType::DimensionsTag;
   using BaseType              = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;

   using MeshType              = Mesh< MeshConfig >;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using CellTopology          = typename MeshTraitsType::CellTopology;
   using StorageArrayType      = typename EntityTraitsType::StorageArrayType;

   using InitializerType       = MeshInitializer< MeshConfig >;
   using EntityInitializerType = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType     = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using PointArrayType        = typename MeshTraitsType::PointArrayType;
   using SeedType              = MeshEntitySeed< MeshConfig, CellTopology >;

   public:

      void initEntities( InitializerType& initializer, PointArrayType& points, CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         initializer.template setNumberOfEntities< DimensionsTag::value >( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
            EntityInitializerType::initEntity( mesh.template getEntity< DimensionsTag::value >( i ), i, cellSeeds[ i ], initializer );
         cellSeeds.reset();

         BaseType::initEntities( initializer, points, mesh );
      }

      using BaseType::findEntitySeedIndex;

      // TODO: this is unused - should be moved to MeshIntegrityChecker
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
               if( this->getMesh().getCell( cell ).getVertexIndex( i ) == - 1 )
               {
                  std::cerr << "The cell number " << cell << " does not have properly set vertex index number " << i << "." << std::endl;
                  return false;
               }
               if( this->getMesh().getCell( cell ).getVertexIndex( i ) >= numberOfVertices )
               {
                  std::cerr << "The cell number " << cell << " does not have properly set vertex index number " << i
                       << ". The index " << this->getMesh().getCell( cell ).getVertexIndex( i )
                       << "is higher than the number of all vertices ( " << numberOfVertices
                       << " )." << std::endl;
                  return false;
               }
            }
         return true;
      }
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned off
 */
template< typename MeshConfig,
          typename DimensionTag >
class MeshInitializerLayer< MeshConfig,
                            DimensionsTag,
                            true,
                            false >
   : public MeshInitializerLayer< MeshConfig,
                                  typename DimensionsTag::Decrement >
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using BaseType              = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;

   using MeshType              = Mesh< MeshConfig >;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using CellTopology          = typename MeshTraitsType::CellTopology;
   using StorageArrayType      = typename EntityTraitsType::StorageArrayType;

   using InitializerType       = MeshInitializer< MeshConfig >;
   using EntityInitializerType = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntitySeedArrayType   = typename EntityTraitsType::SeedArrayType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using PointArrayType        = typename MeshTraitsType::PointArrayType;
   using SeedType              = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SeedIndexedSet = typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType;

   public:

      GlobalIndexType getNumberOfEntities( InitializerType& initializer, MeshType& mesh )
      {
         using SubentitySeedsCreator = MeshSubentitySeedsCreator< MeshConfig, MeshDimensionsTag< CellTopology::dimensions >, DimensionsTag >;
         std::set< typename SeedIndexedSet::key_type > seedSet;

         for( GlobalIndexType i = 0; i < mesh.getNumberOfCells(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.getCell( i ), i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
               seedSet.insert( subentitySeeds[ j ] );
         }

         return seedSet.size();
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed )
      {
         return this->seedsIndexedSet.insert( seed );
      }

      void initEntities( InitializerType& initializer, PointArrayType& points, MeshType& mesh )
      {
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         const GlobalIndexType numberOfEntities = getNumberOfEntities( initializer, mesh );
         initializer.template setNumberOfEntities< DimensionsTag::value >( numberOfEntities );

         using SubentitySeedsCreator = MeshSubentitySeedsCreator< MeshConfig, MeshDimensionsTag< CellTopology::dimensions >, DimensionsTag >;
         for( GlobalIndexType i = 0; i < mesh.getNumberOfCells(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.getCell( i ), i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
            {
               auto& seed = subentitySeeds[ j ];
               if( this->seedsIndexedSet.count( seed ) == 0 ) {
                  const GlobalIndexType entityIndex = this->seedsIndexedSet.insert( seed );
                  EntityInitializerType::initEntity( mesh.template getEntity< DimensionsTag::value >( entityIndex ), entityIndex, seed, initializer );
               }
            }
         }

         EntityInitializerType::initSuperentities( initializer, mesh );
         this->seedsIndexedSet.clear();

         BaseType::initEntities( initializer, points, mesh );
      }

//      void createEntityReferenceOrientations()
//      {
//         BaseType::createEntityReferenceOrientations();
//      }

      using BaseType::getReferenceOrientation;
      using ReferenceOrientationType = typename EntityTraitsType::ReferenceOrientationType;
      const ReferenceOrientationType& getReferenceOrientation( DimensionsTag, GlobalIndexType index ) const {}

   private:
      SeedIndexedSet seedsIndexedSet;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned on
 */
template< typename MeshConfig,
          typename DimensionTag >
class MeshInitializerLayer< MeshConfig,
                            DimensionTag,
                            true,
                            true >
   : public MeshInitializerLayer< MeshConfig,
                                  typename DimensionsTag::Decrement >
{
   using BaseType                      = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;
   using MeshType                      = Mesh< MeshConfig >;
   using MeshTraitsType                = typename MeshType::MeshTraitsType;

   using EntityTraitsType              = typename MeshType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology                = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType               = typename MeshTraitsType::GlobalIndexType;
   using CellTopology                  = typename MeshTraitsType::CellTopology;

   using InitializerType               = MeshInitializer< MeshConfig >;
   using EntityInitializerType         = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType             = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType                = typename MeshTraitsType::LocalIndexType;
   using PointArrayType                = typename MeshTraitsType::PointArrayType;
   using EntityArrayType               = typename EntityTraitsType::StorageArrayType;
   using SeedArrayType                 = typename EntityTraitsType::SeedArrayType;
   using SeedType                      = MeshEntitySeed< MeshConfig, EntityTopology >;
   using ReferenceOrientationType      = typename EntityTraitsType::ReferenceOrientationType;
   using ReferenceOrientationArrayType = typename EntityTraitsType::ReferenceOrientationArrayType;

   public:

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef MeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionsTag >  SubentitySeedsCreator;
         //std::cout << " Creating mesh entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //std::cout << "  Creating mesh entities from cell number " << i << " : " << cellSeeds[ i ] << std::endl;
            typedef typename SubentitySeedsCreator::SubentitySeedArray SubentitySeedArray;
            SubentitySeedArray subentytiSeeds( SubentitySeedsCreator::create( cellSeeds[ i ] ) );
            for( LocalIndexType j = 0; j < subentytiSeeds.getSize(); j++ )
            {
               //std::cout << "Creating subentity seed no. " << j << " : " << subentytiSeeds[ j ] << std::endl;
               //MeshEntitySeed< MeshConfigBase< CellTopology >, EntityTopology >& entitySeed = subentytiSeeds[ j ];
               this->seedsIndexedSet.insert( subentytiSeeds[ j ] );
            }
         }
//         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed )
      {
         return this->seedsIndexedSet.insert( seed );
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points, const CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         createEntitySeedsFromCellSeeds( cellSeeds );
         createEntityReferenceOrientations();
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         initializer.template setNumberOfEntities< DimensionsTag::value >( this->seedsIndexedSet.getSize() );
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //std::cout << "  Initiating entity " << i << std::endl;
            EntityInitializerType::initEntity( mesh.template getEntity< DimensionsTag::value >( i ), i, seedsArray[ i ], initializer );
         }
         this->seedsIndexedSet.reset();

         BaseType::initEntities( initializer, points, cellSeeds, mesh );
      }

      using BaseType::getReferenceOrientation;
      const ReferenceOrientationType& getReferenceOrientation( DimensionsTag, GlobalIndexType index ) const
      {
         return this->referenceOrientations[ index ];
      }

      void createEntityReferenceOrientations()
      {
         //std::cout << " Creating entity reference orientations with " << DimensionsTag::value << " dimensions ... " << std::endl;
         SeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         this->referenceOrientations.setSize( seedsArray.getSize() );
         for( GlobalIndexType i = 0; i < seedsArray.getSize(); i++ )
         {
            //std::cout << "  Creating reference orientation for entity " << i << std::endl;
            this->referenceOrientations[ i ] = ReferenceOrientationType( seedsArray[ i ] );
         }
//         BaseType::createEntityReferenceOrientations();
		}

   private:

      using SeedIndexedSet = typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType;
      SeedIndexedSet seedsIndexedSet;
      ReferenceOrientationArrayType referenceOrientations;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename MeshConfig,
          typename DimensionTag >
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
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = typename MeshType::MeshTraitsType;
   using DimensionsTag         = MeshDimensionsTag< 0 >;

   using EntityTraitsType      = typename MeshType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;

   using InitializerType       = MeshInitializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType        = typename MeshTraits< MeshConfig >::LocalIndexType;
   using PointArrayType        = typename MeshTraits< MeshConfig >::PointArrayType;
   using EntityArrayType       = typename EntityTraitsType::StorageArrayType;
   using EntityInitializerType = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using SeedType              = MeshEntitySeed< MeshConfig, EntityTopology >;

   public:

      GlobalIndexType findEntitySeedIndex( const SeedType& seed )
      {
         return seed.getCornerIds()[ 0 ];
      }

      void initEntities( InitializerType& initializer, PointArrayType& points, MeshType& mesh )
      {
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         initializer.template setNumberOfEntities< 0 >( points.getSize() );
         for( GlobalIndexType i = 0; i < points.getSize(); i++ )
            EntityInitializerType::initEntity( mesh.template getEntity< 0 >( i ), i, points[ i ], initializer );
         points.reset();

         EntityInitializerType::initSuperentities( initializer, mesh );
      }

      // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
      void findEntitySeedIndex() {}

//      void createEntityReferenceOrientations() const {}

      using ReferenceOrientationType = typename EntityTraitsType::ReferenceOrientationType;
      const ReferenceOrientationType& getReferenceOrientation( DimensionsTag, GlobalIndexType index ) const {}
};

} // namespace Meshes
} // namespace TNL
