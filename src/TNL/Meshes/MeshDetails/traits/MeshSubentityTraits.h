/***************************************************************************
                          MeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
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

#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/SharedArray.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimap.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology >
class MeshEntityOrientation;

template< typename MeshConfig,
          typename EntityTopology,
          int Dimension >
class MeshSubentityTraits
{
public:
   static constexpr bool storageEnabled = MeshConfig::subentityStorage( EntityTopology(), Dimensions );
   static constexpr bool orientationEnabled = MeshConfig::subentityOrientationStorage( EntityTopology(), Dimensions );

   using GlobalIndexType   = typename MeshConfig::GlobalIndexType;
   using LocalIndexType    = typename MeshConfig::LocalIndexType;
   using SubentityTopology = typename MeshEntityTraits< MeshConfig, Dimensions >::EntityTopology;
   using SubentityType     = typename MeshEntityTraits< MeshConfig, Dimensions >::EntityType;
   using Seed              = MeshEntitySeed< MeshConfig, SubentityTopology >;
   using Orientation       = MeshEntityOrientation< MeshConfig, SubentityTopology >;


   static constexpr int count = MeshSubtopology< EntityTopology, Dimensions >::count;

   /****
    * Type of container for storing of the subentities indices.
    */
   using StorageNetworkType     = StaticEllpackIndexMultimap< count, GlobalIndexType, Devices::Host, LocalIndexType >;
   using SubentityAccessorType  = typename StorageNetworkType::ValuesAccessorType;

   // static arrays used by MeshEntitySeed etc.
   using IdArrayType            = Containers::StaticArray< count, GlobalIndexType >;
   using SeedArrayType          = Containers::StaticArray< count, Seed >;

   // orientation and its accessor
   using OrientationArrayType   = Containers::StaticArray< count, Orientation >;
   using IdPermutationArrayType = Containers::StaticArray< count, LocalIndexType >;

   template< LocalIndexType subentityIndex,
             LocalIndexType subentityVertexIndex >
   struct Vertex
   {
      enum { index = tnlSubentityVertex< EntityTopology,
                                         SubentityTopology,
                                         subentityIndex,
                                         subentityVertexIndex>::index };
   };

   static_assert( EntityTopology::dimensions > Dimensions, "You try to create subentities traits where subentity dimensions are not smaller than the entity dimensions." );
};

} // namespace Meshes
} // namespace TNL
