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
#include <TNL/Meshes/MeshDetails/initializer/EntitySeed.h>

/*
 * How this beast works:
 *
 * The algorithm is optimized for memory requirements. Therefore, the mesh is
 * not allocated at once, but by parts (by dimensions). The flow is roughly the
 * following:
 *
 *  - Initialize vertices by copying their physical coordinates from the input
 *    array, deallocate the input array of points.
 *  - Initialize cells by copying their subvertex indices from cell seeds (but
 *    other subentity indices are left uninitialized), deallocate cell seeds
 *    (the cells will be used later).
 *  - For all dimensions D from (cell dimension - 1) to 1:
 *     - Create intermediate entity seeds, count the number of entities with
 *       current dimension.
 *     - Set their subvertex indices. Create an indexed set of entity seeds.
 *     - For all superdimensions S > D:
 *        - Iterate over entities with dimension S and initialize their
 *          subentity indices with dimension D. Inverse mapping (D->S) is
 *          recorded in the process.
 *        - For entities with dimension D, initialize their superentity indices
 *          with dimension S.
 *     - Deallocate all intermediate data structures.
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
          typename DimensionTag >
class InitializerLayer;


template< typename MeshConfig >
class Initializer
   : public InitializerLayer< MeshConfig, typename MeshTraits< MeshConfig >::DimensionTag >
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
      using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

      template< int Dimension, int Subdimension >
      using SubentityMatrixRowsCapacitiesType = typename MeshTraitsType::template SubentityMatrixType< Dimension, Subdimension >::RowsCapacitiesType;

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
      void initSubentityMatrix( const NeighborCountsArray& capacities, GlobalIndexType subentitiesCount = 0 )
      {
         if( Subdimension == 0 )
            subentitiesCount = mesh->template getEntitiesCount< 0 >();
         auto& matrix = mesh->template getSubentitiesMatrix< Dimension, Subdimension >();
         matrix.setDimensions( capacities.getSize(), subentitiesCount );
         matrix.setRowCapacities( capacities );
         mesh->template setSubentitiesCounts< Dimension, Subdimension >( capacities );
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
};

/****
 * Mesh initializer layer for cells
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig, typename MeshTraits< MeshConfig >::DimensionTag >
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
   using NeighborCountsArray   = typename MeshTraitsType::NeighborCountsArray;

   public:

      void initEntities( InitializerType& initializer, CellSeedArrayType& cellSeeds, MeshType& mesh )
      {
         //std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
         initializer.template setEntitiesCount< DimensionTag::value >( cellSeeds.getSize() );

         NeighborCountsArray capacities( cellSeeds.getSize() );

         for( LocalIndexType i = 0; i < capacities.getSize(); i++ )
            capacities[ i ] = cellSeeds[ i ].getCornersCount();

         EntityInitializerType::initSubvertexMatrix( capacities, initializer );

         for( GlobalIndexType i = 0; i < cellSeeds.getSize(); i++ )
            EntityInitializerType::initEntity( i, cellSeeds[ i ], initializer );
         cellSeeds.reset();

         BaseType::initEntities( initializer, mesh );
      }

      using BaseType::findEntitySeedIndex;
};

/****
 * Mesh initializer layer for mesh entities other than cells
 */
template< typename MeshConfig,
          typename DimensionTag >
class InitializerLayer
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
   using NeighborCountsArray   = typename MeshTraitsType::NeighborCountsArray;

   public:

      GlobalIndexType getEntitiesCount( InitializerType& initializer, MeshType& mesh )
      {
         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, typename MeshTraitsType::CellTopology, DimensionTag >;
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

         NeighborCountsArray capacities( numberOfEntities );
         int vertexCount = MeshTraitsType::template SubentityTraits< EntityTopology, 0 >::count;
         capacities.setValue( vertexCount );
         EntityInitializerType::initSubvertexMatrix( capacities, initializer );

         using SubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, typename MeshTraitsType::CellTopology, DimensionTag >;
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

   private:
      SeedIndexedSet seedsIndexedSet;
};

/****
 * Mesh initializer layer for vertices
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig, DimensionTag< 0 > >
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
};

} // namespace Meshes
} // namespace TNL
