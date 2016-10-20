/***************************************************************************
                          MeshEntityTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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
struct MeshEntityTopology
{
   static_assert( Dimensions <= MeshConfig::meshDimensions, "There are no entities with dimension higher than the mesh dimension." );
   using Topology = typename MeshSubtopology< typename MeshConfig::CellTopology, Dimensions >::Topology;
};

template< typename MeshConfig >
struct MeshEntityTopology< MeshConfig, MeshConfig::CellTopology::dimensions >
{
   using Topology = typename MeshConfig::CellTopology;
};

} // namespace Meshes
} // namespace TNL
