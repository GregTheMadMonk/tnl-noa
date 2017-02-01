/***************************************************************************
                          MeshHexahedronTopology.h  -  description
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

#include <TNL/Meshes/Topologies/MeshQuadrilateralTopology.h>

namespace TNL {
namespace Meshes {

struct MeshHexahedronTopology
{
   static constexpr int dimension = 3;

   static String getType()
   {
      return "MeshHexahedronTopology";
   }
};

template<>
struct MeshSubtopology< MeshHexahedronTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static constexpr int count = 8;
};

template<>
struct MeshSubtopology< MeshHexahedronTopology, 1 >
{
   typedef MeshEdgeTopology Topology;

   static constexpr int count = 12;
};

template<>
struct MeshSubtopology< MeshHexahedronTopology, 2 >
{
   typedef MeshQuadrilateralTopology Topology;

   static constexpr int count = 6;
};

/****
 * Indexing of the vertices follows the VTK file format
 *
 *        7+---------------------------+6
 *        /|                          /|
 *       / |                         / |
 *      /  |                        /  |
 *     /   |                       /   |
 *   4+---------------------------+5   |
 *    |    |                      |    |
 *    |    |                      |    |
 *    |   3+----------------------|----+2
 *    |   /                       |   /
 *    |  /                        |  /
 *    | /                         | /
 *    |/                          |/
 *   0+---------------------------+1
 *
 *
 * The edges are indexed as follows:
 *
 *         +---------------------------+
 *        /|           10             /|
 *     11/ |                         / |
 *      /  |                        /9 |
 *     /  7|                       /   |6
 *    +---------------------------+    |
 *    |    |        8             |    |
 *    |    |                      |    |
 *    |    +----------------------|----+
 *   4|   /           2           |5  /
 *    | 3/                        |  /
 *    | /                         | /1
 *    |/                          |/
 *    +---------------------------+
 *                 0
 *
 * The faces are indexed as follows (the indexed are positioned to
 * the opposite corners of given face):
 *
 *         +---------------------------+
 *        /|5                        3/|
 *       /4|                         /2|
 *      /  |                        /  |
 *     /   |                     5 /   |
 *    +---------------------------+    |
 *    |1   |                      |    |
 *    |    |3                     |    |
 *    |    +----------------------|----+
 *    |   /                       |  0/
 *    |  /                        |  /
 *    |4/                         |2/
 *    |/0                        1|/
 *    +---------------------------+
 *
 */

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  0, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  0, 1> { enum { index = 1 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  1, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  1, 1> { enum { index = 2 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  2, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  2, 1> { enum { index = 3 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  3, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  3, 1> { enum { index = 0 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  4, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  4, 1> { enum { index = 4 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  5, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  5, 1> { enum { index = 5 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  6, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  6, 1> { enum { index = 6 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  7, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  7, 1> { enum { index = 7 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  8, 0> { enum { index = 4 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  8, 1> { enum { index = 5 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  9, 0> { enum { index = 5 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology,  9, 1> { enum { index = 6 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology, 10, 0> { enum { index = 6 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology, 10, 1> { enum { index = 7 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology, 11, 0> { enum { index = 7 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshEdgeTopology, 11, 1> { enum { index = 4 }; };


template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 0, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 0, 1> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 0, 2> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 0, 3> { enum { index = 3 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 1, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 1, 1> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 1, 2> { enum { index = 5 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 1, 3> { enum { index = 4 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 2, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 2, 1> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 2, 2> { enum { index = 6 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 2, 3> { enum { index = 5 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 3, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 3, 1> { enum { index = 3 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 3, 2> { enum { index = 7 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 3, 3> { enum { index = 6 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 4, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 4, 1> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 4, 2> { enum { index = 4 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 4, 3> { enum { index = 7 }; };

template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 5, 0> { enum { index = 4 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 5, 1> { enum { index = 5 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 5, 2> { enum { index = 6 }; };
template<> struct SubentityVertexMap< MeshHexahedronTopology, MeshQuadrilateralTopology, 5, 3> { enum { index = 7 }; };

} // namespace Meshes
} // namespace TNL
