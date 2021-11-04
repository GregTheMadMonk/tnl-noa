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
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          int Dimension >
class MeshSubentityTraits
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   static_assert( EntityTopology::dimension > Dimension, "Subentity dimension must be smaller than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::subentityStorage( EntityTopology::dimension, Dimension );
   static constexpr int count = Topologies::Subtopology< EntityTopology, Dimension >::count;

   using SubentityTopology = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityTopology;
   using SubentityType     = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityType;

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
