/***************************************************************************
                          MeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
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

#include <TNL/Containers/StaticArray.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimap.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology >
class MeshEntityOrientation;
template< typename MeshConfig, typename EntityTopology >
class MeshEntitySeed;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          int Dimension >
class MeshSubentityTraits
{
public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   static_assert( EntityTopology::dimension > Dimension, "Subentity dimension must be smaller than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::subentityStorage( EntityTopology(), Dimension );
   static constexpr bool orientationEnabled = MeshConfig::subentityOrientationStorage( EntityTopology(), Dimension );
   static constexpr int count = Topologies::Subtopology< EntityTopology, Dimension >::count;

   using GlobalIndexType   = typename MeshConfig::GlobalIndexType;
   using LocalIndexType    = typename MeshConfig::LocalIndexType;
   using SubentityTopology = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityTopology;
   using SubentityType     = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityType;
   using Seed              = MeshEntitySeed< MeshConfig, SubentityTopology >;
   using Orientation       = MeshEntityOrientation< MeshConfig, SubentityTopology >;

   /****
    * Type of container for storing of the subentities indices.
    */
   // TODO: write general operator= for different SliceSize and remove the '32' here
   using StorageNetworkType     = StaticEllpackIndexMultimap< count, GlobalIndexType, Device, LocalIndexType, 32 >;
   using SubentityAccessorType  = typename StorageNetworkType::ValuesAccessorType;

   // static array used in MeshSubentitySeedCreator
   using SeedArrayType          = Containers::StaticArray< count, Seed >;

   // orientation and its accessor
   using OrientationArrayType   = Containers::StaticArray< count, Orientation >;
   using IdPermutationArrayType = Containers::StaticArray< count, LocalIndexType >;

   template< LocalIndexType subentityIndex,
             LocalIndexType subentityVertexIndex >
   struct Vertex
   {
      static constexpr int index = Topologies::SubentityVertexMap<
                  EntityTopology,
                  SubentityTopology,
                  subentityIndex,
                  subentityVertexIndex >::index;
   };
};

} // namespace Meshes
} // namespace TNL
