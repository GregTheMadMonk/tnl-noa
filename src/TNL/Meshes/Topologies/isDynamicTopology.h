#pragma once

#include <type_traits>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/Topologies/Vertex.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

template < typename Topology, int SubDimension, typename = int >
struct SubtopologyHasCount : std::false_type
{};

template < typename Topology, int SubDimension >
struct SubtopologyHasCount< Topology, SubDimension, decltype(Subtopology< Topology, SubDimension >::count, int{}) > : std::true_type
{};

template< typename Topology, int Dimension = Topology::dimension >
struct isDynamicTopology
{
   enum : bool { value = !SubtopologyHasCount< Topology, Dimension - 1 >::value ||
                         isDynamicTopology< Topology, Dimension - 1 >::value };
};

template<>
struct isDynamicTopology< Vertex, 0 > : std::false_type
{};

template< typename Topology >
struct isDynamicTopology< Topology, 1 >
{
   enum : bool { value = !SubtopologyHasCount< Topology, 0 >::value };
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL