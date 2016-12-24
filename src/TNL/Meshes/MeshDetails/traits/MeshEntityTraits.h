/***************************************************************************
                          MeshEntityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#include <TNL/Containers/Array.h>
#include <TNL/Containers/IndexedSet.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeed.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntityReferenceOrientation;

template< typename MeshConfig,
          typename EntityDimensionTag,
          typename SuperDimensionTag = DimensionTag< MeshConfig::meshDimension > >
class MeshEntityOrientationNeeded
{
   using SuperentityTopology = typename MeshTraits< MeshConfig >::template EntityTraits< SuperDimensionTag::value >::EntityTopology;

   static constexpr bool previousSuperDimensionValue = MeshEntityOrientationNeeded< MeshConfig, EntityDimensionTag, typename SuperDimensionTag::Decrement >::value;
   static constexpr bool thisSuperDimensionValue = MeshTraits< MeshConfig >::template SubentityTraits< SuperentityTopology, EntityDimensionTag::value >::orientationEnabled;

public:
   static constexpr bool value = ( previousSuperDimensionValue || thisSuperDimensionValue );
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
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );

   static constexpr bool storageEnabled = MeshConfig::entityStorage( Dimension );
   static constexpr bool orientationNeeded = MeshEntityOrientationNeeded< MeshConfig, DimensionTag< Dimension > >::value;

   using GlobalIndexType               = typename MeshConfig::GlobalIndexType;
   using LocalIndexType                = typename MeshConfig::LocalIndexType;
   using EntityTopology                = typename MeshEntityTopology< MeshConfig, DimensionTag< Dimension > >::Topology;

   using EntityType                    = MeshEntity< MeshConfig, EntityTopology >;
   using SeedType                      = MeshEntitySeed< MeshConfig, EntityTopology >;
   using ReferenceOrientationType      = MeshEntityReferenceOrientation< MeshConfig, EntityTopology >;

   using StorageArrayType              = Containers::Array< EntityType, Devices::Host, GlobalIndexType >;
   using SeedIndexedSetType            = Containers::IndexedSet< typename SeedType::KeyType, GlobalIndexType >;
   using ReferenceOrientationArrayType = Containers::Array< ReferenceOrientationType, Devices::Host, GlobalIndexType >;
};

} // namespace Meshes
} // namespace TNL
