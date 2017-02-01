/***************************************************************************
                          MeshEntityTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#include <type_traits>

#include <TNL/Meshes/DimensionTag.h>

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
struct SubentityVertexMap
{
};


template< typename MeshConfig,
          typename DimensionTag >
struct MeshEntityTopology
{
   static_assert( DimensionTag::value <= MeshConfig::meshDimension, "There are no entities with dimension higher than the mesh dimension." );
   using Topology = typename MeshSubtopology< typename MeshConfig::CellTopology, DimensionTag::value >::Topology;
};

template< typename MeshConfig >
struct MeshEntityTopology< MeshConfig, DimensionTag< MeshConfig::CellTopology::dimension > >
{
   using Topology = typename MeshConfig::CellTopology;
};


/* Helper struct to determine if one topology is compatible with another one. */
template< typename Supertopology, typename Subtopology >
struct is_compatible_topology
{
   static_assert( Supertopology::dimension >= Subtopology::dimension,
                  "wrong order of topologies in template parameters" );
   static constexpr bool value = std::is_same< typename MeshSubtopology< Supertopology, Subtopology::dimension >::Topology,
                                               Subtopology >::value;
};

template< typename Supertopology >
struct is_compatible_topology< Supertopology, Supertopology >
{
   static constexpr bool value = true;
};

} // namespace Meshes
} // namespace TNL
