/***************************************************************************
                          Initializer.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/initializer/EntityInitializer.h>
#include <TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>
#include <TNL/Meshes/MeshDetails/MeshEntityReferenceOrientation.h>
#include <TNL/Meshes/MeshDetails/initializer/EntitySeed.h>

/*
 * How this beast works:
 *
 * The algorithm is optimized for memory requirements. Therefore, the mesh is
 * not allocated at once, but by parts (by dimensions). The flow is roughly the
 * following:
 *
 *  - Allocate cells and set their subvertex indices (but not other subentity
 *    indices), deallocate cell seeds (the cells will be used later).
 *  - For all dimensions D from (cell dimension - 1) to 1:
 *     - Create intermediate entity seeds, count the number of entities with
 *       current dimension.
 *     - Allocate entities and set their subvertex indices. Create an indexed
 *       set of entity seeds and reference orientations (if applicable).
 *     - For all superdimensions S > D:
 *        - Iterate over entities with dimension S and initialize their
 *          subentity indices with dimension D. Inverse mapping (D->S) is
 *          recorded in the process.
 *        - For entities with dimension D, initialize their superentity indices
 *          with dimension S.
 *     - Deallocate all intermediate data structures.
 *  - Allocate vertices and set their physical coordinates, deallocate input
 *    array of points.
 *  - For all superdimensions S > 0, repeat the same steps as above.
 *
 * Optimization notes:
 *   - Recomputing the seed key involves sorting all subvertex indices, but the
 *     cost is negligible compared to memory consumption incurred by storing
 *     both the key and original seed in the indexed set.
 *   - Since std::set and std::map don't provide a shrink_to_fit method like
 *     std::vector, these dynamic structures should be kept as local variables
 *     if possible. This is probably the only way to be sure that the unused
 *     space is not wasted.
 */

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename DimensionTag,
          bool EntityStorage =
             MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::storageEnabled,
          bool EntityReferenceOrientationStorage =
             MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::orientationNeeded >
class InitializerLayer;


template< typename MeshConfig >
class Initializer
   : public InitializerLayer< MeshConfig,
                              typename MeshTraits< MeshConfig >::DimensionTag >
{
   protected:
      // must be declared before its use in expression with decltype()
      Mesh< MeshConfig >* mesh;

   public:
      using MeshType          = Mesh< MeshConfig >;
      using MeshTraitsType    = MeshTraits< MeshConfig >;
      using DimensionTag      = Meshes::DimensionTag< MeshTraitsType::meshDimension >;
      using BaseType          = InitializerLayer< MeshConfig, DimensionTag >;
      using PointArrayType    = typename MeshTraitsType::PointArrayType;
      using CellSeedArrayType = typename MeshTraitsType::CellSeedArrayType;
      using GlobalIndexType   = typename MeshTraitsType::GlobalIndexType;


      Initializer()
      : mesh( 0 )
      {}

      // The points and cellSeeds arrays will be reset when not needed to save memory.
      void createMesh( PointArrayType& points,
                       CellSeedArrayType& cellSeeds,
                       MeshType& mesh )
      {
         this->mesh = &mesh;
         BaseType::initEntities( *this, points, cellSeeds, mesh );
         // set pointers from entities into the subentity and superentity storage networks
         EntityStorageRebinder< Mesh< MeshConfig > >::exec( mesh );
      }

      template< typename Entity, typename GlobalIndex >
      void setEntityIndex( Entity& entity, const GlobalIndex& index )
      {
         entity.setIndex( index );
      }

      template< int Dimension >
      void setEntitiesCount( const GlobalIndexType& entitiesCount )
      {
         //std::cout << "Setting number of entities with " << Dimension << " dimension to " << entitiesCount << std::endl;
         mesh->template setEntitiesCount< Dimension >( entitiesCount );
      }

      template< int Subdimension, typename EntityType, typename LocalIndex, typename GlobalIndex >
      void
      setSubentityIndex( const EntityType& entity, const GlobalIndex& entityIndex, const LocalIndex& localIndex, const GlobalIndex& globalIndex )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimension, Subdimension >().getValues( entityIndex )[ localIndex ] = globalIndex;
      }

      template< int Subdimension, typename EntityType, typename LocalIndex, typename GlobalIndex >
      GlobalIndex
      getSubentityIndex( const EntityType& entity, const GlobalIndex& entityIndex, const LocalIndex& localIndex )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         return mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimension, Subdimension >().getValues( entityIndex )[ localIndex ];
      }

      template< typename EntityType, typename GlobalIndex >
      auto
      getSubvertices( const EntityType& entity, const GlobalIndex& entityIndex )
         -> decltype( this->mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimension, 0 >().getValues( 0 ) )
      {
         // The mesh entities are not yet bound to the storage network at this point,
         // so we operate directly on the storage.
         return mesh->template getSubentityStorageNetwork< EntityType::EntityTopology::dimension, 0 >().getValues( entityIndex );
      }

      template< typename EntityTopology, int Superdimension >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >::StorageNetworkType&
      meshSuperentityStorageNetwork()
      {
         return mesh->template getSuperentityStorageNetwork< EntityTopology::dimension, Superdimension >();
      }

      static void
      setVertexPoint( typename MeshType::Vertex& vertex, const typename MeshType::PointType& point )
      {
         vertex.setPoint( point );
      }


      template< typename SubDimensionTag, typename MeshEntity >
      static typename MeshTraitsType::template SubentityTraits< typename MeshEntity::EntityTopology, SubDimensionTag::value >::OrientationArrayType&
      subentityOrientationsArray( MeshEntity& entity )
      {
         return entity.template subentityOrientationsArray< SubDimensionTag::value >();
      }

      template< typename DimensionTag >
      const MeshEntityReferenceOrientation< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionTag::value >::EntityTopology >&
      getReferenceOrientation( GlobalIndexType index ) const
      {
         return BaseType::getReferenceOrientation( DimensionTag(), index );
      }
};

/****
 * Mesh initializer layer for cells
 *  - entities storage must turned on (cells must always be stored )
 *  - entities orientation does not make sense for cells => it is turned off
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig,
                        typename MeshTraits< MeshConfig >::DimensionTag,
                        true,
                        false >
   : public InitializerLayer< MeshConfig,
                              typename MeshTraits< MeshConfig >::DimensionTag::Decrement >
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using DimensionTag          = typename MeshTraitsType::DimensionTag;
   using BaseType              = InitializerLayer< MeshConfig, typename DimensionTag::Decrement >;

   using MeshType              = Mesh< MeshConfig >;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;

   using InitializerType       = Initializer< MeshConfig >;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType     = typename MeshTraitsType::CellSeedArrayType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using PointArrayType        = typename MeshTraitsType::PointArrayType;

   public:

      void initEntities( InitializerType& initializer, PointArrayType& points, CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         initializer.template setEntitiesCount< DimensionTag::value >( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
            EntityInitializerType::initEntity( mesh.template getEntity< DimensionTag::value >( i ), i, cellSeeds[ i ], initializer );
         cellSeeds.reset();

         BaseType::initEntities( initializer, points, mesh );
      }

      using BaseType::findEntitySeedIndex;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities storage is turned on
 * - entities orientation storage is turned off
 */
template< typename MeshConfig,
          typename DimensionTag >
class InitializerLayer< MeshConfig,
                        DimensionTag,
                        true,
                        false >
   : public InitializerLayer< MeshConfig,
                              typename DimensionTag::Decrement >
{
   using BaseType              = InitializerLayer< MeshConfig, typename DimensionTag::Decrement >;
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;

   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;

   using InitializerType       = Initializer< MeshConfig >;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using PointArrayType        = typename MeshTraitsType::PointArrayType;
   using SeedType              = EntitySeed< MeshConfig, EntityTopology >;
   using SeedIndexedSet        = typename MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::SeedIndexedSetType;
   using SeedSet               = typename MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::SeedSetType;

   public:

      GlobalIndexType getEntitiesCount( InitializerType& initializer, MeshType& mesh )
      {
         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         SeedSet seedSet;

         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.template getEntity< MeshType::getMeshDimension() >( i ), i ) );
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
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         const GlobalIndexType numberOfEntities = getEntitiesCount( initializer, mesh );
         initializer.template setEntitiesCount< DimensionTag::value >( numberOfEntities );

         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.template getEntity< MeshType::getMeshDimension() >( i ), i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
            {
               auto& seed = subentitySeeds[ j ];
               const auto pair = this->seedsIndexedSet.try_insert( seed );
               const GlobalIndexType& entityIndex = pair.first;
               if( pair.second ) {
                  // insertion took place, initialize the entity
                  EntityInitializerType::initEntity( mesh.template getEntity< DimensionTag::value >( entityIndex ), entityIndex, seed, initializer );
               }
            }
         }

         EntityInitializerType::initSuperentities( initializer, mesh );
         this->seedsIndexedSet.clear();

         BaseType::initEntities( initializer, points, mesh );
      }

      using BaseType::getReferenceOrientation;
      using ReferenceOrientationType = typename EntityTraitsType::ReferenceOrientationType;
      const ReferenceOrientationType& getReferenceOrientation( DimensionTag, GlobalIndexType index ) const {}

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
class InitializerLayer< MeshConfig,
                        DimensionTag,
                        true,
                        true >
   : public InitializerLayer< MeshConfig,
                              typename DimensionTag::Decrement >
{
   using BaseType                      = InitializerLayer< MeshConfig, typename DimensionTag::Decrement >;
   using MeshType                      = Mesh< MeshConfig >;
   using MeshTraitsType                = typename MeshType::MeshTraitsType;

   using EntityTraitsType              = typename MeshType::template EntityTraits< DimensionTag::value >;
   using EntityTopology                = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType               = typename MeshTraitsType::GlobalIndexType;

   using InitializerType               = Initializer< MeshConfig >;
   using EntityInitializerType         = EntityInitializer< MeshConfig, EntityTopology >;
   using LocalIndexType                = typename MeshTraitsType::LocalIndexType;
   using PointArrayType                = typename MeshTraitsType::PointArrayType;
   using SeedType                      = EntitySeed< MeshConfig, EntityTopology >;
   using SeedIndexedSet                = typename MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::SeedIndexedSetType;
   using SeedSet                       = typename MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::SeedSetType;
   using ReferenceOrientationType      = typename EntityTraitsType::ReferenceOrientationType;
   using ReferenceOrientationArrayType = typename EntityTraitsType::ReferenceOrientationArrayType;

   public:

      GlobalIndexType getEntitiesCount( InitializerType& initializer, MeshType& mesh )
      {
         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         SeedSet seedSet;

         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.template getEntity< MeshType::getMeshDimension() >( i ), i ) );
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
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         const GlobalIndexType numberOfEntities = getEntitiesCount( initializer, mesh );
         initializer.template setEntitiesCount< DimensionTag::value >( numberOfEntities );
         this->referenceOrientations.setSize( numberOfEntities );

         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.getSubvertices( mesh.template getEntity< MeshType::getMeshDimension() >( i ), i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
            {
               auto& seed = subentitySeeds[ j ];
               const auto pair = this->seedsIndexedSet.try_insert( seed );
               const GlobalIndexType& entityIndex = pair.first;
               if( pair.second ) {
                  // insertion took place, initialize the entity
                  EntityInitializerType::initEntity( mesh.template getEntity< DimensionTag::value >( entityIndex ), entityIndex, seed, initializer );
                  this->referenceOrientations[ entityIndex ] = ReferenceOrientationType( seed );
               }
            }
         }

         EntityInitializerType::initSuperentities( initializer, mesh );
         this->seedsIndexedSet.clear();
         this->referenceOrientations.reset();

         BaseType::initEntities( initializer, points, mesh );
      }

      using BaseType::getReferenceOrientation;
      const ReferenceOrientationType& getReferenceOrientation( DimensionTag, GlobalIndexType index ) const
      {
         return this->referenceOrientations[ index ];
      }

   private:
      SeedIndexedSet seedsIndexedSet;
      ReferenceOrientationArrayType referenceOrientations;
};

/****
 * Mesh initializer layer for entities not being stored
 */
template< typename MeshConfig,
          typename DimensionTag >
class InitializerLayer< MeshConfig,
                        DimensionTag,
                        false,
                        false >
   : public InitializerLayer< MeshConfig,
                              typename DimensionTag::Decrement >
{};

/****
 * Mesh initializer layer for vertices
 * - vertices must always be stored
 * - their orientation does not make sense
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig,
                        DimensionTag< 0 >,
                        true,
                        false >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = typename MeshType::MeshTraitsType;
   using DimensionTag          = Meshes::DimensionTag< 0 >;

   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;

   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType        = typename MeshTraits< MeshConfig >::LocalIndexType;
   using PointArrayType        = typename MeshTraits< MeshConfig >::PointArrayType;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
   using SeedType              = EntitySeed< MeshConfig, EntityTopology >;

   public:

      GlobalIndexType findEntitySeedIndex( const SeedType& seed )
      {
         return seed.getCornerIds()[ 0 ];
      }

      void initEntities( InitializerType& initializer, PointArrayType& points, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         initializer.template setEntitiesCount< 0 >( points.getSize() );
         for( GlobalIndexType i = 0; i < points.getSize(); i++ )
            EntityInitializerType::initEntity( mesh.template getEntity< 0 >( i ), i, points[ i ], initializer );
         points.reset();

         EntityInitializerType::initSuperentities( initializer, mesh );
      }

      // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
      void findEntitySeedIndex() {}

      using ReferenceOrientationType = typename EntityTraitsType::ReferenceOrientationType;
      const ReferenceOrientationType& getReferenceOrientation( DimensionTag, GlobalIndexType index ) const {}
};

} // namespace Meshes
} // namespace TNL