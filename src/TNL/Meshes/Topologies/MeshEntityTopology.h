/***************************************************************************
                          MeshEntityTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes{

template< typename MeshEntityTopology,
          int SubentityDimension >
struct MeshSubtopology
{
};

template< typename MeshEntityTopology,
          typename SubentityTopology,
          int SubentityIndex,
          int SubentityVertexIndex >
struct tnlSubentityVertex;


template< typename MeshConfig,
          int Dimensions >
class MeshEntityTopology
{
   public:

   typedef typename MeshSubtopology< typename MeshConfig::CellTopology,
                                        Dimensions >::Topology Topology;
};

template< typename MeshConfig >
class MeshEntityTopology< MeshConfig,
                             MeshConfig::CellTopology::dimensions >
{
   public:

   typedef typename MeshConfig::CellTopology Topology;
};

} // namespace Meshes
} // namespace TNL