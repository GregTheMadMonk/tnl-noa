#pragma once

#include <TNL/Meshes/Topologies/Polygon.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

struct Polyhedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Polyhedron, 0 >
{
   typedef Vertex Topology;
};

template<>
struct Subtopology< Polyhedron, 1 >
{
   typedef Edge Topology;
};

template<>
struct Subtopology< Polyhedron, 2 >
{
   typedef Polygon Topology;
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL