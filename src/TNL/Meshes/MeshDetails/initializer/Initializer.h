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
 *  - Allocate vertices and set their physical coordinates, deallocate input
 *    array of points.
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
      Mesh< MeshConfig >* mesh = nullptr;

   public:
      using MeshType          = Mesh< MeshConfig >;
      using MeshTraitsType    = MeshTraits< MeshConfig >;
      using DimensionTag      = Meshes::DimensionTag< MeshTraitsType::meshDimension >;
      using BaseType          = InitializerLayer< MeshConfig, DimensionTag >;
      using PointArrayType    = typename MeshTraitsType::PointArrayType;
      using CellSeedArrayType = typename MeshTraitsType::CellSeedArrayType;
      using GlobalIndexType   = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType    = typename MeshTraitsType::LocalIndexType;


      // The points and cellSeeds arrays will be reset when not needed to save memory.
      void createMesh( PointArrayType& points,
                       CellSeedArrayType& cellSeeds,
                       MeshType& mesh )
      {
         // copy points
         mesh.template setEntitiesCount< 0 >( points.getSize() );
         mesh.getPoints().swap( points );
         points.reset();

         this->mesh = &mesh;
         BaseType::initEntities( *this, cellSeeds, mesh );
      }

      template< int Dimension, int Subdimension >
      void initSubentityMatrix( const GlobalIndexType entitiesCount, GlobalIndexType subentitiesCount = 0 )
      {
         if( Subdimension == 0 )
            subentitiesCount = mesh->template getEntitiesCount< 0 >();
         auto& matrix = mesh->template getSubentitiesMatrix< Dimension, Subdimension >();
         matrix.setDimensions( entitiesCount, subentitiesCount );
         using EntityTraitsType    = typename MeshTraitsType::template EntityTraits< Dimension >;
         using EntityTopology      = typename EntityTraitsType::EntityTopology;
         using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >;
         constexpr int count = SubentityTraitsType::count;
         typename std::decay_t<decltype(matrix)>::RowsCapacitiesType capacities( entitiesCount );
         capacities.setValue( count );
         matrix.setRowCapacities( capacities );
      }

      template< int Dimension >
      void setEntitiesCount( const GlobalIndexType entitiesCount )
      {
         mesh->template setEntitiesCount< Dimension >( entitiesCount );
      }

      template< int Dimension, int Subdimension >
      void
      setSubentityIndex( const GlobalIndexType entityIndex, const LocalIndexType localIndex, const GlobalIndexType globalIndex )
      {
         mesh->template getSubentitiesMatrix< Dimension, Subdimension >().getRow( entityIndex ).setElement( localIndex, globalIndex, true );
      }

      template< int Dimension >
      auto
      getSubvertices( const GlobalIndexType entityIndex )
         -> decltype( this->mesh->template getSubentitiesMatrix< Dimension, 0 >().getRow( 0 ) )
      {
         return mesh->template getSubentitiesMatrix< Dimension, 0 >().getRow( entityIndex );
      }

      template< int Dimension, int Superdimension >
      auto
      getSuperentitiesCountsArray()
         -> decltype( this->mesh->template getSuperentitiesCountsArray< Dimension, Superdimension >() )
      {
         return mesh->template getSuperentitiesCountsArray< Dimension, Superdimension >();
      }

      template< int Dimension, int Superdimension >
      auto
      getSuperentitiesMatrix()
         -> decltype( this->mesh->template getSuperentitiesMatrix< Dimension, Superdimension >() )
      {
         return mesh->template getSuperentitiesMatrix< Dimension, Superdimension >();
      }


      template< int Dimension, int Subdimension >
      auto
      subentityOrientationsArray( const GlobalIndexType entityIndex )
         -> decltype( this->mesh->template subentityOrientationsArray< Dimension, Subdimension >( entityIndex ) )
      {
         return mesh->template subentityOrientationsArray< Dimension, Subdimension >( entityIndex );
      }

      template< typename DimensionTag >
      const MeshEntityReferenceOrientation< MeshConfig, typename MeshTraitsType::template EntityTraits< DimensionTag::value >::EntityTopology >&
      getReferenceOrientation( const GlobalIndexType index ) const
      {
         return BaseType::getReferenceOrientation( DimensionTag(), index );
      }
};

/****
 * Mesh initializer layer for cells
 *  - entities orientation does not make sense for cells => it is turned off
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig,
                        typename MeshTraits< MeshConfig >::DimensionTag,
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
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;

   using InitializerType       = Initializer< MeshConfig >;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
   using CellSeedArrayType     = typename MeshTraitsType::CellSeedArrayType;

   public:

      void initEntities( InitializerType& initializer, CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         initializer.template setEntitiesCount< DimensionTag::value >( cellSeeds.getSize() );
         initializer.template initSubentityMatrix< DimensionTag::value, 0 >( cellSeeds.getSize() );
         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
            EntityInitializerType::initEntity( i, cellSeeds[ i ], initializer );
         cellSeeds.reset();

         BaseType::initEntities( initializer, mesh );
      }

      using BaseType::findEntitySeedIndex;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities orientation storage is turned off
 */
template< typename MeshConfig,
          typename DimensionTag >
class InitializerLayer< MeshConfig,
                        DimensionTag,
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
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;

   using InitializerType       = Initializer< MeshConfig >;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
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
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.template getSubvertices< MeshType::getMeshDimension() >( i ) );
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

      void initEntities( InitializerType& initializer, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         const GlobalIndexType numberOfEntities = getEntitiesCount( initializer, mesh );
         initializer.template setEntitiesCount< DimensionTag::value >( numberOfEntities );
         EntityInitializerType::initSubvertexMatrix( numberOfEntities, initializer );

         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.template getSubvertices< MeshType::getMeshDimension() >( i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
            {
               auto& seed = subentitySeeds[ j ];
               const auto pair = this->seedsIndexedSet.try_insert( seed );
               const GlobalIndexType& entityIndex = pair.first;
               if( pair.second ) {
                  // insertion took place, initialize the entity
                  EntityInitializerType::initEntity( entityIndex, seed, initializer );
               }
            }
         }

         EntityInitializerType::initSuperentities( initializer, mesh );
         this->seedsIndexedSet.clear();

         BaseType::initEntities( initializer, mesh );
      }

      using BaseType::getReferenceOrientation;
      void getReferenceOrientation( DimensionTag, GlobalIndexType index ) const {}

   private:
      SeedIndexedSet seedsIndexedSet;
};

/****
 * Mesh initializer layer for other mesh entities than cells
 * - entities orientation storage is turned on
 */
template< typename MeshConfig,
          typename DimensionTag >
class InitializerLayer< MeshConfig,
                        DimensionTag,
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
   using LocalIndexType                = typename MeshTraitsType::LocalIndexType;

   using InitializerType               = Initializer< MeshConfig >;
   using EntityInitializerType         = EntityInitializer< MeshConfig, EntityTopology >;
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
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.template getSubvertices< MeshType::getMeshDimension() >( i ) );
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

      void initEntities( InitializerType& initializer, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         const GlobalIndexType numberOfEntities = getEntitiesCount( initializer, mesh );
         initializer.template setEntitiesCount< DimensionTag::value >( numberOfEntities );
         EntityInitializerType::initSubvertexMatrix( numberOfEntities, initializer );
         this->referenceOrientations.resize( numberOfEntities );

         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Meshes::DimensionTag< MeshType::getMeshDimension() >, DimensionTag >;
         for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ )
         {
            auto subentitySeeds = SubentitySeedsCreator::create( initializer.template getSubvertices< MeshType::getMeshDimension() >( i ) );
            for( LocalIndexType j = 0; j < subentitySeeds.getSize(); j++ )
            {
               auto& seed = subentitySeeds[ j ];
               const auto pair = this->seedsIndexedSet.try_insert( seed );
               const GlobalIndexType& entityIndex = pair.first;
               if( pair.second ) {
                  // insertion took place, initialize the entity
                  EntityInitializerType::initEntity( entityIndex, seed, initializer );
                  this->referenceOrientations[ entityIndex ] = ReferenceOrientationType( seed );
               }
            }
         }

         EntityInitializerType::initSuperentities( initializer, mesh );
         this->seedsIndexedSet.clear();
         this->referenceOrientations.clear();
         this->referenceOrientations.shrink_to_fit();

         BaseType::initEntities( initializer, mesh );
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
 * Mesh initializer layer for vertices
 * - their orientation does not make sense
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig,
                        DimensionTag< 0 >,
                        false >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = typename MeshType::MeshTraitsType;
   using DimensionTag          = Meshes::DimensionTag< 0 >;

   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityTopology        = typename EntityTraitsType::EntityTopology;
   using GlobalIndexType       = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType        = typename MeshTraits< MeshConfig >::LocalIndexType;

   using InitializerType       = Initializer< MeshConfig >;
   using EntityInitializerType = EntityInitializer< MeshConfig, EntityTopology >;
   using SeedType              = EntitySeed< MeshConfig, EntityTopology >;

   public:

      GlobalIndexType findEntitySeedIndex( const SeedType& seed )
      {
         return seed.getCornerIds()[ 0 ];
      }

      void initEntities( InitializerType& initializer, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         EntityInitializerType::initSuperentities( initializer, mesh );
      }

      // This method is due to 'using BaseType::getReferenceOrientation;' in the derived class.
      void getReferenceOrientation( DimensionTag, GlobalIndexType index ) const {}
};

} // namespace Meshes
} // namespace TNL
