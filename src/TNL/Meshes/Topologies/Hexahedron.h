/***************************************************************************
                          Hexahedron.h  -  description
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

#include <TNL/Meshes/Topologies/Quadrilateral.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

struct Hexahedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Hexahedron, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 8;
};

template<>
struct Subtopology< Hexahedron, 1 >
{
   typedef Edge Topology;

   static constexpr int count = 12;
};

template<>
struct Subtopology< Hexahedron, 2 >
{
   typedef Quadrilateral Topology;

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

template<> struct SubentityVertexMap< Hexahedron, Edge,  0, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  0, 1> { enum { index = 1 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  1, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  1, 1> { enum { index = 2 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  2, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  2, 1> { enum { index = 3 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  3, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  3, 1> { enum { index = 0 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  4, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  4, 1> { enum { index = 4 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  5, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  5, 1> { enum { index = 5 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  6, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  6, 1> { enum { index = 6 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  7, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  7, 1> { enum { index = 7 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  8, 0> { enum { index = 4 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  8, 1> { enum { index = 5 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  9, 0> { enum { index = 5 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  9, 1> { enum { index = 6 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge, 10, 0> { enum { index = 6 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge, 10, 1> { enum { index = 7 }; };

template<> struct SubentityVertexMap< Hexahedron, Edge, 11, 0> { enum { index = 7 }; };
template<> struct SubentityVertexMap< Hexahedron, Edge, 11, 1> { enum { index = 4 }; };


template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 0, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 0, 1> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 0, 2> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 0, 3> { enum { index = 3 }; };

template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 1, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 1, 1> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 1, 2> { enum { index = 5 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 1, 3> { enum { index = 4 }; };

template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 2, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 2, 1> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 2, 2> { enum { index = 6 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 2, 3> { enum { index = 5 }; };

template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 3, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 3, 1> { enum { index = 3 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 3, 2> { enum { index = 7 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 3, 3> { enum { index = 6 }; };

template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 4, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 4, 1> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 4, 2> { enum { index = 4 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 4, 3> { enum { index = 7 }; };

template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 5, 0> { enum { index = 4 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 5, 1> { enum { index = 5 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 5, 2> { enum { index = 6 }; };
template<> struct SubentityVertexMap< Hexahedron, Quadrilateral, 5, 3> { enum { index = 7 }; };

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
