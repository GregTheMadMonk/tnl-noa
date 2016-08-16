/***************************************************************************
                          MeshTriangleTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Topologies/MeshEdgeTopology.h>

namespace TNL {
namespace Meshes {

struct MeshTriangleTopology
{
   static const int dimensions = 2;
};


template<>
struct MeshSubtopology< MeshTriangleTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static const int count = 3;
};

template<>
struct MeshSubtopology< MeshTriangleTopology, 1 >
{
   typedef MeshEdgeTopology Topology;

   static const int count = 3;
};


template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTriangleTopology, MeshEdgeTopology, 2, 1> { enum { index = 1 }; };

} // namespace Meshes
} // namespace TNL