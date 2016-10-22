/***************************************************************************
                          MeshSuperentityStorageInitializer.h  -  description
                             -------------------
    begin                : Feb 27, 2014
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

#include <set>
#include <map>

#include <TNL/Meshes/MeshDimensionsTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag,
          bool SuperentityStorage = MeshSuperentityTraits< MeshConfig, EntityTopology, DimensionTag::value >::storageEnabled >
class MeshSuperentityStorageInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializer
   : public MeshSuperentityStorageInitializerLayer< MeshConfig, EntityTopology, MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag,
                                              true >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                    EntityTopology,
                                                    typename DimensionsTag::Decrement >
{
   using BaseType = MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                            EntityTopology,
                                                            typename DimensionsTag::Decrement >;

   static const int Dimensions = DimensionsTag::value;
   using EntityDimensions          = MeshDimensionsTag< EntityTopology::dimensions >;
   using EntityType                = MeshEntity< MeshConfig, EntityTopology >;

   using MeshTraitsType            = MeshTraits< MeshConfig >;
   using GlobalIdArrayType         = typename MeshTraitsType::GlobalIdArrayType;
   using GlobalIndexType           = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType            = typename MeshTraitsType::LocalIndexType;
   using MeshInitializerType       = MeshInitializer< MeshConfig >;
   using SuperentityTraitsType     = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimensions >;
   using SuperentityStorageNetwork = typename SuperentityTraitsType::StorageNetworkType;

   public:
      using BaseType::addSuperentity;

      void addSuperentity( DimensionsTag, GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
      {
         //std::cout << "Adding superentity with " << DimensionsTag::value << " dimensions of entity with " << EntityDimensions::value << " dimensions: entityIndex = " << entityIndex << ", superentityIndex = " << superentityIndex << std::endl;
         auto& indexSet = this->dynamicStorageNetwork[ entityIndex ];
         Assert( indexSet.count( superentityIndex ) == 0,
                    std::cerr << "Superentity " << superentityIndex << " with dimensions " << DimensionsTag::value
                              << " of entity " << entityIndex << " with dimensions " << EntityDimensions::value
                              << " has been already added. This is probably a bug in the mesh initializer." << std::endl; );
         indexSet.insert( superentityIndex );
      }

      using BaseType::initSuperentities;
      void initSuperentities( MeshInitializerType& meshInitializer )
      {
         if( ! dynamicStorageNetwork.empty() ) {
            GlobalIndexType maxEntityIndex = 0;
            for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ ) {
               if( it->first > maxEntityIndex )
                  maxEntityIndex = it->first;
            }

            Assert( (size_t) maxEntityIndex == dynamicStorageNetwork.size() - 1,
                       std::cerr << "Superentities for some entities are missing." << std::endl; );

            /****
             * Network initializer
             */
            SuperentityStorageNetwork& superentityStorageNetwork = meshInitializer.template meshSuperentityStorageNetwork< EntityTopology, DimensionsTag >();
            superentityStorageNetwork.setKeysRange( maxEntityIndex + 1 );
            typename SuperentityStorageNetwork::ValuesAllocationVectorType storageNetworkAllocationVector;
            storageNetworkAllocationVector.setSize( maxEntityIndex + 1 );
            for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ )
               storageNetworkAllocationVector[ it->first ] = it->second.size();
            superentityStorageNetwork.allocate( storageNetworkAllocationVector );

            GlobalIndexType entityIndex = 0;
            for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ ) {
               auto superentitiesIndices = superentityStorageNetwork.getValues( it->first );
               LocalIndexType i = 0;
               for( auto v_it = it->second.cbegin(); v_it != it->second.cend(); v_it++ )
                  superentitiesIndices[ i++ ] = *v_it;

               EntityType& entity = meshInitializer.template meshEntitiesArray< EntityDimensions::value >()[ entityIndex ];
               meshInitializer.template bindSuperentitiesStorageNetwork< DimensionsTag::value >( entity, superentityStorageNetwork.getValues( entityIndex++ ) );
            }

            dynamicStorageNetwork.clear();
         }

         BaseType::initSuperentities( meshInitializer );
      }

   private:
      using DynamicIndexSet = std::set< GlobalIndexType >;
      std::map< GlobalIndexType, DynamicIndexSet > dynamicStorageNetwork;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                              EntityTopology,
                                              DimensionsTag,
                                              false >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                    EntityTopology,
                                                    typename DimensionsTag::Decrement >
{
   using BaseType = MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                            EntityTopology,
                                                            typename DimensionsTag::Decrement >;
   using MeshInitializerType = MeshInitializer< MeshConfig >;

public:
   // Necessary due to 'using BaseType::...;' in the derived classes.
   void addSuperentity() {}
   void initSuperentities( MeshInitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionTag< EntityTopology::dimensions >,
                                          true >
{
   using MeshInitializerType = MeshInitializer< MeshConfig >;

public:
   // Necessary due to 'using BaseType::...;' in the derived classes.
   void addSuperentity() {}
   void initSuperentities( MeshInitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionTag< EntityTopology::dimensions >,
                                          false >
{
   using MeshInitializerType = MeshInitializer< MeshConfig >;

public:
   // Necessary due to 'using BaseType::...;' in the derived classes.
   void addSuperentity() {}
   void initSuperentities( MeshInitializerType& ) {}
};

} // namespace Meshes
} // namespace TNL
