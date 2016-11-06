/***************************************************************************
                          MeshTriangleTopology.h  -  description
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

#include <TNL/Meshes/Topologies/MeshEdgeTopology.h>

namespace TNL {
namespace Meshes {

struct MeshTriangleTopology
{
   static constexpr int dimensions = 2;

   static String getType()
   {
      return "MeshTriangleTopology";
   }
};


template<>
struct MeshSubtopology< MeshTriangleTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static constexpr int count = 3;
};

template<>
struct MeshSubtopology< MeshTriangleTopology, 1 >
{
   typedef MeshEdgeTopology Topology;

   static constexpr int count = 3;
};


template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 1> { enum { index = 1 }; };

} // namespace Meshes
} // namespace TNL
