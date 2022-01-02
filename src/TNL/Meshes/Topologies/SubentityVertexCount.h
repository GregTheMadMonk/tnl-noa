#pragma once

#include <TNL/Meshes/Topologies/SubentityVertexMap.h>

namespace TNL {
namespace Meshes{
namespace Topologies {

template< typename EntityTopology,
          typename SubentityTopology,
          int SubentityIndex >
struct SubentityVertexCount
{
   static constexpr int count = Subtopology< SubentityTopology, 0 >::count;
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL