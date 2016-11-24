/***************************************************************************
                          MeshTetrahedronTopology.h  -  description
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

#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>

namespace TNL {
namespace Meshes {

struct MeshTetrahedronTopology
{
   static constexpr int dimension = 3;

   static String getType()
   {
      return "MeshTetrahedronTopology";
   }
};

template<>
struct MeshSubtopology< MeshTetrahedronTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static constexpr int count = 4;
};

template<>
struct MeshSubtopology< MeshTetrahedronTopology, 1 >
{
   typedef MeshEdgeTopology Topology;

   static constexpr int count = 6;
};

template<>
struct MeshSubtopology< MeshTetrahedronTopology, 2 >
{
   typedef MeshTriangleTopology Topology;

   static constexpr int count = 4;
};


template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 2, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 3, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 4, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshEdgeTopology, 5, 1> { enum { index = 3 }; };


template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 0, 2> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 1, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 1> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 2, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 1> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< MeshTetrahedronTopology, MeshTriangleTopology, 3, 2> { enum { index = 3 }; };

} // namespace Meshes
} // namespace TNL
