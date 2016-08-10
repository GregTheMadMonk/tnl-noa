/***************************************************************************
                          MeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Arrays/StaticArray.h>
#include <TNL/Arrays/SharedArray.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology > class MeshEntityOrientation;

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class MeshSubentityTraits
{
   public:
      static const bool storageEnabled = MeshConfig::subentityStorage( EntityTopology(), Dimensions );
      static const bool orientationEnabled = MeshConfig::subentityOrientationStorage( EntityTopology(), Dimensions );

      typedef typename MeshConfig::GlobalIndexType                                GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                 LocalIndexType;
      typedef MeshSubtopology< EntityTopology, Dimensions >                    Subtopology;
      typedef typename Subtopology::Topology                                      SubentityTopology;
      typedef MeshEntity< MeshConfig, SubentityTopology >                      SubentityType;
      typedef MeshEntitySeed< MeshConfig, SubentityTopology >                  Seed;
      typedef MeshEntityOrientation< MeshConfig, SubentityTopology >           Orientation;


      static const int count = Subtopology::count;

      typedef Arrays::StaticArray< count, GlobalIndexType >              StorageArrayType;
      typedef Arrays::SharedArray< GlobalIndexType,
                                      Devices::Host,
                                      LocalIndexType >                      AccessArrayType;
      typedef Arrays::StaticArray< count, GlobalIndexType >              IdArrayType;
      typedef Arrays::StaticArray< count, SubentityType >                SubentityContainerType;
      typedef Arrays::StaticArray< count, Seed >                         SeedArrayType;
      typedef Arrays::StaticArray< count, Orientation >                  OrientationArrayType;
      typedef Arrays::StaticArray< count, LocalIndexType >               IdPermutationArrayType;

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
