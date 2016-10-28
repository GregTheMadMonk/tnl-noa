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
             MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::orientationNeeded >
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
      using MeshType          = Mesh< MeshConfig >;
      using MeshTraitsType    = MeshTraits< MeshConfig >;
      using DimensionsTag     = MeshDimensionsTag< MeshTraitsType::meshDimensions >;
      using BaseType          = MeshInitializerLayer< MeshConfig, DimensionsTag >;
      using PointArrayType    = typename MeshTraitsType::PointArrayType;
      using CellSeedArrayType = typename MeshTraitsType::CellSeedArrayType;
      using GlobalIndexType   = typename MeshTraitsType::GlobalIndexType;


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
         if( verbose ) std::cout << "======= Starting mesh initiation ========" << std::endl;
         this->mesh = &mesh;

         if( verbose ) std::cout << "========= Creating entity seeds =============" << std::endl;
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );

         if( verbose ) std::cout << "========= Creating entity reference orientations =============" << std::endl;
         BaseType::createEntityReferenceOrientations();

         if( verbose ) std::cout << "====== Initiating entities ==============" << std::endl;
         BaseType::initEntities( *this, points, cellSeeds, mesh );

         return true;
      }

      template< int Dimensions >
      bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
      {
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

      template< typename DimensionTag >
      MeshSuperentityStorageInitializer< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionTag::value >::EntityTopology >&
      getSuperentityInitializer()
      {
         return BaseType::getSuperentityInitializer( DimensionTag() );
      }


      template< typename DimensionsTag >
      const MeshEntityReferenceOrientation< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionsTag::value >::EntityTopology >&
      getReferenceOrientation( GlobalIndexType index ) const
      {
         return BaseType::getReferenceOrientation( DimensionsTag(), index );
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
   using MeshTraitsType               = MeshTraits< MeshConfig >;
   static constexpr int Dimensions    = MeshTraitsType::meshDimensions;
   using DimensionsTag                = MeshDimensionsTag< Dimensions >;
   using BaseType                     = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;

   using MeshType                     = Mesh< MeshConfig >;
   using EntityTraitsType             = typename MeshTraitsType::template EntityTraits< Dimensions >;
   using EntityTopology               = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType              = typename MeshTraitsType::GlobalIndexType;
   using CellTopology                 = typename MeshTraitsType::CellTopology;
   using StorageArrayType             = typename EntityTraitsType::StorageArrayType;

   using InitializerType              = MeshInitializer< MeshConfig >;
   using EntityInitializerType        = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType            = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType               = typename MeshTraitsType::LocalIndexType;
   using PointArrayType               = typename MeshTraitsType::PointArrayType;
   using SeedType                     = MeshEntitySeed< MeshConfig, CellTopology >;
   using SuperentityInitializerType   = MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >;

   public:

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         BaseType::createEntitySeedsFromCellSeeds( cellSeeds );
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points, const CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         initializer.template setNumberOfEntities< Dimensions >( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
         {
            //std::cout << "  Initiating entity " << i << std::endl;
            EntityInitializerType::initEntity( mesh.template getEntity< Dimensions >( i ), i, cellSeeds[i], initializer );
         }
         /***
          * There are no superentities in this layer storing mesh cells.
          */

         BaseType::initEntities( initializer, points, mesh );
      }

      using BaseType::findEntitySeedIndex;
      GlobalIndexType findEntitySeedIndex( const SeedType& seed ) const
      {
         return this->seedsIndexedSet.find( seed );
      }

      using BaseType::getSuperentityInitializer;
      SuperentityInitializerType& getSuperentityInitializer( DimensionTag )
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

   private:

      using SeedIndexedSet = typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
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
   using MeshTraitsType               = MeshTraits< MeshConfig >;
   using BaseType                     = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;

   using MeshType                     = Mesh< MeshConfig >;
   using EntityTraitsType             = typename MeshTraitsType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology               = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType              = typename MeshTraitsType::GlobalIndexType;
   using CellTopology                 = typename MeshTraitsType::CellTopology;
   using StorageArrayType             = typename EntityTraitsType::StorageArrayType;

   using InitializerType              = MeshInitializer< MeshConfig >;
   using EntityInitializerType        = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using EntitySeedArrayType          = typename EntityTraitsType::SeedArrayType;
   using CellSeedArrayType            = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType               = typename MeshTraitsType::LocalIndexType;
   using PointArrayType               = typename MeshTraitsType::PointArrayType;
   using SeedType                     = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SuperentityInitializerType   = MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >;

   public:

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef MeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionTag::value << " dimensions ... " << std::endl;
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
      SuperentityInitializerType& getSuperentityInitializer( DimensionTag )
      {
         return this->superentityInitializer;
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points, MeshType& mesh )
      {
         //std::cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
         initializer.template setNumberOfEntities< DimensionsTag::value >( this->seedsIndexedSet.getSize() );
         EntitySeedArrayType seedsArray;
         seedsArray.setSize( this->seedsIndexedSet.getSize() );
         this->seedsIndexedSet.toArray( seedsArray );
         for( GlobalIndexType i = 0; i < this->seedsIndexedSet.getSize(); i++ )
         {
            //std::cout << "  Initiating entity " << i << std::endl;
            EntityInitializerType::initEntity( mesh.template getEntity< DimensionsTag::value >( i ), i, seedsArray[ i ], initializer );
         }
         this->seedsIndexedSet.reset();

         this->superentityInitializer.initSuperentities( initializer );

         BaseType::initEntities( initializer, points, mesh );
      }

      void createEntityReferenceOrientations()
      {
         BaseType::createEntityReferenceOrientations();
      }

   private:

      using SeedIndexedSet = typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
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
   using BaseType                       = MeshInitializerLayer< MeshConfig, typename DimensionsTag::Decrement >;
   using MeshType                       = Mesh< MeshConfig >;
   using MeshTraitsType                 = typename MeshType::MeshTraitsType;

   using EntityTraitsType               = typename MeshType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology                 = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType                = typename MeshTraitsType::GlobalIndexType;
   using CellTopology                   = typename MeshTraitsType::CellTopology;

   using InitializerType                = MeshInitializer< MeshConfig >;
   using EntityInitializerType          = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType              = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType                 = typename MeshTraitsType::LocalIndexType;
   using PointArrayType                 = typename MeshTraitsType::PointArrayType;
   using EntityArrayType                = typename EntityTraitsType::StorageArrayType;
   using SeedArrayType                  = typename EntityTraitsType::SeedArrayType;
   using SeedType                       = MeshEntitySeed< MeshConfig, EntityTopology >;
   using SuperentityInitializerType     = MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >;
   using ReferenceOrientationType       = typename EntityTraitsType::ReferenceOrientationType;
   using ReferenceOrientationArrayType  = typename EntityTraitsType::ReferenceOrientationArrayType;

   public:

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds )
      {
         typedef MeshSubentitySeedsCreator< MeshConfig, CellTopology, DimensionTag >  SubentitySeedsCreator;
         //cout << " Creating mesh entities with " << DimensionTag::value << " dimensions ... " << std::endl;
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
      SuperentityInitializerType& getSuperentityInitializer( DimensionTag )
      {
         return this->superentityInitializer;
      }

      void initEntities( InitializerType& initializer, const PointArrayType& points, MeshType& mesh )
      {
         //cout << " Initiating entities with " << DimensionsTag::value << " dimensions ... " << std::endl;
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

         this->superentityInitializer.initSuperentities( initializer );

         BaseType::initEntities( initializer, points, mesh );
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
         BaseType::createEntityReferenceOrientations();
		}

   private:

      using SeedIndexedSet = typename MeshEntityTraits< MeshConfig, DimensionsTag::value >::SeedIndexedSetType;
      SeedIndexedSet seedsIndexedSet;
      SuperentityInitializerType superentityInitializer;
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
   using MeshType                       = Mesh< MeshConfig >;
   using MeshTraitsType                 = typename MeshType::MeshTraitsType;
   using DimensionsTag                  = MeshDimensionsTag< 0 >;

   using EntityTraitsType               = typename MeshType::template EntityTraits< DimensionsTag::value >;
   using EntityTopology                 = typename EntityTraitsType::EntityTopology;
   using ContainerType                  = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType                = typename ContainerType::IndexType;

   using CellTopology                   = typename MeshTraitsType::CellTopology;

   using InitializerType                = MeshInitializer< MeshConfig >;
   using CellSeedArrayType              = typename MeshTraits< MeshConfig >::CellSeedArrayType;
   using LocalIndexType                 = typename MeshTraits< MeshConfig >::LocalIndexType;
   using PointArrayType                 = typename MeshTraits< MeshConfig >::PointArrayType;
   using EntityArrayType                = typename EntityTraitsType::StorageArrayType;
   using EntityInitializerType          = MeshEntityInitializer< MeshConfig, EntityTopology >;
   using SuperentityInitializerType     = MeshSuperentityStorageInitializer< MeshConfig, EntityTopology >;

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

      void createEntitySeedsFromCellSeeds( const CellSeedArrayType& cellSeeds ) {}

      void initEntities( InitializerType& initializer, const PointArrayType& points, MeshType& mesh )
      {
         initializer.template setNumberOfEntities< 0 >( points.getSize() );
         for( GlobalIndexType i = 0; i < points.getSize(); i++ )
            EntityInitializerType::setVertexPoint( mesh.template getEntity< 0 >( i ), points[i], initializer );

         superentityInitializer.initSuperentities( initializer );
      }

      // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
      void findEntitySeedIndex() const {}

      SuperentityInitializerType& getSuperentityInitializer( DimensionsTag )
      {
         return this->superentityInitializer;
      }

      void createEntityReferenceOrientations() const {}

      void getReferenceOrientation() const {}

   private:

      SuperentityInitializerType superentityInitializer;

      MeshType* mesh;
};

} // namespace Meshes
} // namespace TNL
