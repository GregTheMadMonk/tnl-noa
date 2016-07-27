/***************************************************************************
                          tnlMeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Arrays/StaticArray.h>
#include <TNL/Arrays/SharedArray.h>
#include <TNL/mesh/tnlMeshEntity.h>
#include <TNL/mesh/config/tnlMeshConfigBase.h>
#include <TNL/mesh/topologies/tnlMeshEntityTopology.h>

namespace TNL {

template< typename MeshConfig, typename EntityTopology > class tnlMeshEntityOrientation;

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class tnlMeshSubentityTraits
{
   public:
      static const bool storageEnabled = MeshConfig::subentityStorage( EntityTopology(), Dimensions );
      static const bool orientationEnabled = MeshConfig::subentityOrientationStorage( EntityTopology(), Dimensions );

      typedef typename MeshConfig::GlobalIndexType                                GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                 LocalIndexType;
      typedef tnlMeshSubtopology< EntityTopology, Dimensions >                    Subtopology;
      typedef typename Subtopology::Topology                                      SubentityTopology;
      typedef tnlMeshEntity< MeshConfig, SubentityTopology >                      SubentityType;
      typedef tnlMeshEntitySeed< MeshConfig, SubentityTopology >                  Seed;
      typedef tnlMeshEntityOrientation< MeshConfig, SubentityTopology >           Orientation;


      static const int count = Subtopology::count;

      typedef Arrays::tnlStaticArray< count, GlobalIndexType >              StorageArrayType;
      typedef Arrays::tnlSharedArray< GlobalIndexType,
                                      Devices::Host,
                                      LocalIndexType >                      AccessArrayType;
      typedef Arrays::tnlStaticArray< count, GlobalIndexType >              IdArrayType;
      typedef Arrays::tnlStaticArray< count, SubentityType >                SubentityContainerType;
      typedef Arrays::tnlStaticArray< count, Seed >                         SeedArrayType;
      typedef Arrays::tnlStaticArray< count, Orientation >                  OrientationArrayType;
      typedef Arrays::tnlStaticArray< count, LocalIndexType >               IdPermutationArrayType;

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

} // namespace TNL
