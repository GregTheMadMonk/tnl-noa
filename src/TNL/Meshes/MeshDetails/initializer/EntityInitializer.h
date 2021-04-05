/***************************************************************************
                          EntityInitializer.h  -  description
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

#include <TNL/Meshes/MeshDetails/initializer/EntitySeed.h>
#include <TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class Initializer;

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag,
          // storage in the superentity
          bool SubentityStorage =
               MeshConfig::subentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >::EntityTopology(),
                                             SubdimensionTag::value ),
          // storage in the subentity
          bool SuperentityStorage =
               MeshConfig::superentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< SubdimensionTag::value >::EntityTopology(),
                                               SuperdimensionTag::value ),
          // necessary to disambiguate the stop condition for specializations
          bool valid_dimension = ! std::is_same< SubdimensionTag, SuperdimensionTag >::value >
class EntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology,
          bool SubvertexStorage =
               MeshConfig::subentityStorage( typename MeshTraits< MeshConfig >::template EntityTraits< EntityTopology::dimension >::EntityTopology(), 0 ) >
class EntityInitializer
   : public EntityInitializerLayer< MeshConfig,
                                    DimensionTag< EntityTopology::dimension >,
                                    DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            DimensionTag< EntityTopology::dimension >,
                                            DimensionTag< MeshTraits< MeshConfig >::meshDimension > >;

   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType   = typename MeshTraitsType::LocalIndexType;

   using SeedType         = EntitySeed< MeshConfig, EntityTopology >;
   using InitializerType  = Initializer< MeshConfig >;

public:
   static void initSubvertexMatrix( const GlobalIndexType entitiesCount, InitializerType& initializer )
   {
      initializer.template initSubentityMatrix< EntityTopology::dimension, 0 >( entitiesCount );
   }

   static void initEntity( const GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer )
   {
      // this is necessary if we want to use existing entities instead of intermediate seeds to create subentity seeds
      for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
         initializer.template setSubentityIndex< EntityTopology::dimension, 0 >( entityIndex, i, entitySeed.getCornerIds()[ i ] );
   }
};

template< typename MeshConfig,
          typename EntityTopology >
class EntityInitializer< MeshConfig, EntityTopology, false >
   : public EntityInitializerLayer< MeshConfig,
                                    DimensionTag< EntityTopology::dimension >,
                                    DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;

   using SeedType         = EntitySeed< MeshConfig, EntityTopology >;
   using InitializerType  = Initializer< MeshConfig >;
public:
   static void initSubvertexMatrix( const GlobalIndexType entitiesCount, InitializerType& initializer ) {}
   static void initEntity( const GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer ) {}
};


/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      TRUE                    TRUE
 */
template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              true,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType            = Initializer< MeshConfig >;
   using MeshType                   = typename InitializerType::MeshType;

   using GlobalIndexType            = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType             = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SubentityTraitsType        = typename MeshTraits< MeshConfig >::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology          = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType  = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;
   using SuperentityMatrixType      = typename MeshTraits< MeshConfig >::SuperentityMatrixType;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();
      if( SubdimensionTag::value > 0 )
         meshInitializer.template initSubentityMatrix< SuperdimensionTag::value, SubdimensionTag::value >( superentitiesCount, subentitiesCount );

      // counter for superentities of each subentity
      auto& superentitiesCounts = meshInitializer.template getSuperentitiesCountsArray< SubdimensionTag::value, SuperdimensionTag::value >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );

      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ )
      {
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.template getSubvertices< SuperdimensionTag::value >( superentityIndex ) );
         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex, i, subentityIndex );
            superentitiesCounts[ subentityIndex ]++;
         }
      }

      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< SubdimensionTag::value, SuperdimensionTag::value >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCounts );
      superentitiesCounts.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ )
      {
         for( LocalIndexType i = 0;
              i < mesh.template getSubentitiesCount< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex );
              i++ )
         {
            const GlobalIndexType subentityIndex = mesh.template getSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex, i );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCounts[ subentityIndex ]++, superentityIndex, true );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      TRUE                   FALSE
 */
template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              true,
                              false,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType           = Initializer< MeshConfig >;
   using MeshType                  = typename InitializerType::MeshType;

   using GlobalIndexType           = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType            = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SuperentityTraitsType     = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology       = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();
      if( SubdimensionTag::value > 0 )
         meshInitializer.template initSubentityMatrix< SuperdimensionTag::value, SubdimensionTag::value >( superentitiesCount, subentitiesCount );

      for( GlobalIndexType superentityIndex = 0;
           superentityIndex < mesh.template getEntitiesCount< SuperdimensionTag::value >();
           superentityIndex++ )
      {
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.template getSubvertices< SuperdimensionTag::value >( superentityIndex ) );

         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            meshInitializer.template setSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex, i, subentityIndex );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      FALSE                  TRUE
 */
template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              false,
                              true,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            SubdimensionTag,
                                            typename SuperdimensionTag::Decrement >;
   using InitializerType            = Initializer< MeshConfig >;
   using MeshType                   = typename InitializerType::MeshType;

   using GlobalIndexType            = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType             = typename MeshTraits< MeshConfig >::LocalIndexType;
   using SubentityTraitsType        = typename MeshTraits< MeshConfig >::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology          = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType      = typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology        = typename SuperentityTraitsType::EntityTopology;
   using SubentitySeedsCreatorType  = SubentitySeedsCreator< MeshConfig, SuperdimensionTag, SubdimensionTag >;
   using SuperentityMatrixType      = typename MeshTraits< MeshConfig >::SuperentityMatrixType;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      //std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();

      // counter for superentities of each subentity
      auto& superentitiesCounts = meshInitializer.template getSuperentitiesCountsArray< SubdimensionTag::value, SuperdimensionTag::value >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );

      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ )
      {
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.template getSubvertices< SuperdimensionTag::value >( superentityIndex ) );
         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            superentitiesCounts[ subentityIndex ]++;
         }
      }

      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< SubdimensionTag::value, SuperdimensionTag::value >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCounts );
      superentitiesCounts.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ )
      {
         auto subentitySeeds = SubentitySeedsCreatorType::create( meshInitializer.template getSubvertices< SuperdimensionTag::value >( superentityIndex ) );
         for( LocalIndexType i = 0; i < subentitySeeds.getSize(); i++ )
         {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( subentitySeeds[ i ] );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCounts[ subentityIndex ]++, superentityIndex, true );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SuperdimensionTag,
                              false,
                              false,
                              true >
   : public EntityInitializerLayer< MeshConfig,
                                    SubdimensionTag,
                                    typename SuperdimensionTag::Decrement >
{};

template< typename MeshConfig,
          typename SubdimensionTag,
          bool SubentityStorage,
          bool SuperentityStorage >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SubdimensionTag,
                              SubentityStorage,
                              SuperentityStorage,
                              false >
{
   using InitializerType = Initializer< MeshConfig >;
   using MeshType        = typename InitializerType::MeshType;

public:
   static void initSuperentities( InitializerType& meshInitializer, MeshType& mesh ) {}
};

} // namespace Meshes
} // namespace TNL
