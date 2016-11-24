/***************************************************************************
                          MeshSuperentityStorageInitializer.h  -  description
                             -------------------
    begin                : Feb 27, 2014
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

#include <set>
#include <map>

#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag >
class MeshSuperentityStorageInitializer
{
   using MeshTraitsType            = MeshTraits< MeshConfig >;
   using MeshInitializerType       = MeshInitializer< MeshConfig >;
   using GlobalIndexType           = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType            = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType          = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using EntityTopology            = typename EntityTraitsType::EntityTopology;
   using EntityType                = typename EntityTraitsType::EntityType;
   using SuperentityTraitsType     = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;
   using SuperentityStorageNetwork = typename SuperentityTraitsType::StorageNetworkType;

public:
   void addSuperentity( GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
   {
      //std::cout << "Adding superentity with " << SuperdimensionTag::value << " dimension of entity with " << SubdimensionTag::value << " dimension: entityIndex = " << entityIndex << ", superentityIndex = " << superentityIndex << std::endl;
      auto& indexSet = this->dynamicStorageNetwork[ entityIndex ];
      Assert( indexSet.count( superentityIndex ) == 0,
                 std::cerr << "Superentity " << superentityIndex << " with dimension " << SuperdimensionTag::value
                           << " of entity " << entityIndex << " with dimension " << SubdimensionTag::value
                           << " has been already added. This is probably a bug in the mesh initializer." << std::endl; );
      indexSet.insert( superentityIndex );
   }

   void initSuperentities( MeshInitializerType& meshInitializer )
   {
      Assert( dynamicStorageNetwork.size() > 0,
                 std::cerr << "No superentity indices were collected. This is a bug in the mesh initializer." << std::endl; );
      Assert( (size_t) getMaxSuperentityIndex() == dynamicStorageNetwork.size() - 1,
                 std::cerr << "Superentities for some entities are missing. "
                           << "This is probably a bug in the mesh initializer." << std::endl; );

      SuperentityStorageNetwork& superentityStorageNetwork = meshInitializer.template meshSuperentityStorageNetwork< EntityTopology, SuperdimensionTag::value >();
      Assert( (size_t) superentityStorageNetwork.getKeysRange() == dynamicStorageNetwork.size(),
                 std::cerr << "Sizes of the static and dynamic storage networks don't match. "
                           << "This is probably a bug in the mesh initializer." << std::endl; );

      typename SuperentityStorageNetwork::ValuesAllocationVectorType storageNetworkAllocationVector;
      storageNetworkAllocationVector.setSize( superentityStorageNetwork.getKeysRange() );
      for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ )
         storageNetworkAllocationVector[ it->first ] = it->second.size();
      superentityStorageNetwork.allocate( storageNetworkAllocationVector );

      GlobalIndexType entityIndex = 0;
      for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ ) {
         auto superentitiesIndices = superentityStorageNetwork.getValues( it->first );
         LocalIndexType i = 0;
         for( auto v_it = it->second.cbegin(); v_it != it->second.cend(); v_it++ )
            superentitiesIndices[ i++ ] = *v_it;
      }

      dynamicStorageNetwork.clear();
   }

private:
   using DynamicIndexSet = std::set< GlobalIndexType >;
   std::map< GlobalIndexType, DynamicIndexSet > dynamicStorageNetwork;

   GlobalIndexType getMaxSuperentityIndex()
   {
      GlobalIndexType max = 0;
      for( auto it = dynamicStorageNetwork.cbegin(); it != dynamicStorageNetwork.cend(); it++ ) {
         if( it->first > max )
            max = it->first;
      }
      return max;
   }
};

} // namespace Meshes
} // namespace TNL
