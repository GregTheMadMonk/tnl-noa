#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/Topologies/Vertex.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

/**
 * \brief Type trait for checking if Topology has at least one missing Subtopology< Topology, D > >::count for all D from Topology::dimension - 1 to 0
 */
template< typename Topology, int D = Topology::dimension >
struct IsDynamicTopology
{
   enum : bool { value = !HasCountMember< Subtopology< Topology, D - 1 > >::value ||
                         IsDynamicTopology< Topology, D - 1 >::value };
};

/**
 * \brief Specialization for Vertex Topology
 */
template<>
struct IsDynamicTopology< Vertex, 0 > : std::false_type
{};

/**
 * \brief Specialization for D = 1 to end recursion
 */
template< typename Topology >
struct IsDynamicTopology< Topology, 1 >
{
   enum : bool { value = !HasCountMember< Subtopology< Topology, 0 > >::value };
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL