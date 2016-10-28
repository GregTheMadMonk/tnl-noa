/***************************************************************************
                          MeshEntityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/SharedArray.h>
//#include <TNL/Containers/ConstSharedArray.h>
#include <TNL/Containers/IndexedSet.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeed;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeedKey;
template< typename MeshConfig, typename EntityTopology > class MeshEntityReferenceOrientation;

template< typename MeshConfig,
          typename DimensionTag,
          typename SuperDimensionTag = MeshDimensionTag< MeshConfig::meshDimension > >
class MeshEntityOrientationNeeded
{
   using SuperentityTopology = typename MeshTraits< MeshConfig >::template EntityTraits< SuperDimensionsTag::value >::EntityTopology;

   static constexpr bool previousSuperDimensionsValue = MeshEntityOrientationNeeded< MeshConfig, DimensionsTag, typename SuperDimensionsTag::Decrement >::value;
   static constexpr bool thisSuperDimensionsValue = MeshTraits< MeshConfig >::template SubentityTraits< SuperentityTopology, DimensionsTag::value >::orientationEnabled;

public:
   static constexpr bool value = ( previousSuperDimensionsValue || thisSuperDimensionsValue );
};

template< typename MeshConfig, typename DimensionTag >
class MeshEntityOrientationNeeded< MeshConfig, DimensionTag, DimensionTag >
{
public:
   static constexpr bool value = false;
};


template< typename MeshConfig,
          int Dimension >
class MeshEntityTraits
{
public:
   static constexpr bool storageEnabled = MeshConfig::entityStorage( Dimensions );
   static constexpr bool orientationNeeded = MeshEntityOrientationNeeded< MeshConfig, MeshDimensionsTag< Dimensions > >::value;

   using GlobalIndexType               = typename MeshConfig::GlobalIndexType;
   using LocalIndexType                = typename MeshConfig::LocalIndexType;
   using EntityTopology                = typename MeshEntityTopology< MeshConfig, MeshDimensionsTag< Dimensions > >::Topology;

   using EntityType                    = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                      = MeshEntitySeed< MeshConfig, EntityTopology >;
   using ReferenceOrientationType      = MeshEntityReferenceOrientation< MeshConfig, EntityTopology >;
   using Key                           = MeshEntitySeedKey< MeshConfig, EntityTopology >;

   using StorageArrayType              = Containers::Array< EntityType, Devices::Host, GlobalIndexType >;
   using UniqueContainerType           = Containers::IndexedSet< EntityType, GlobalIndexType, Key >;
   using SeedIndexedSetType            = Containers::IndexedSet< SeedType, GlobalIndexType, Key >;
   using SeedArrayType                 = Containers::Array< SeedType, Devices::Host, GlobalIndexType >;
   using ReferenceOrientationArrayType = Containers::Array< ReferenceOrientationType, Devices::Host, GlobalIndexType >;

   static_assert( 0 <= Dimensions && Dimensions <= MeshConfig::meshDimensions, "invalid dimensions" );
};

} // namespace Meshes
} // namespace TNL
