/***************************************************************************
                          tnlMeshSubentityTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHSUBENTITYTRAITS_H_
#define TNLMESHSUBENTITYTRAITS_H_

#include <core/arrays/tnlStaticArray.h>
#include <core/arrays/tnlSharedArray.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>


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

      typedef tnlStaticArray< count, GlobalIndexType >              StorageArrayType;
      typedef tnlSharedArray< GlobalIndexType,
                              tnlHost,
                              LocalIndexType >                      AccessArrayType;
      typedef tnlStaticArray< count, GlobalIndexType >              IdArrayType;
      typedef tnlStaticArray< count, SubentityType >                SubentityContainerType;
      typedef tnlStaticArray< count, Seed >                         SeedArrayType;
      typedef tnlStaticArray< count, Orientation >                  OrientationArrayType;
      typedef tnlStaticArray< count, LocalIndexType >               IdPermutationArrayType;

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



#endif /* TNLMESHSUBENTITYTRAITS_H_ */
