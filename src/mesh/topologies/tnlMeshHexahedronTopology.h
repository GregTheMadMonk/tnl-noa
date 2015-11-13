/***************************************************************************
                          tnlMeshHexahedronTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHHEXAHEDRONTOPOLOGY_H_
#define TNLMESHHEXAHEDRONTOPOLOGY_H_

#include <mesh/topologies/tnlMeshQuadrilateralTopology.h>

struct tnlMeshHexahedronTopology
{
   static const int dimensions = 3;
};

template<>
struct tnlMeshSubtopology< tnlMeshHexahedronTopology, 0 >
{
   typedef tnlMeshVertexTopology Topology;

   static const int count = 8;
};

template<>
struct tnlMeshSubtopology< tnlMeshHexahedronTopology, 1 >
{
   typedef tnlMeshEdgeTopology Topology;

   static const int count = 12;
};

template<>
struct tnlMeshSubtopology< tnlMeshHexahedronTopology, 2 >
{
   typedef tnlMeshQuadrilateralTopology Topology;

   static const int count = 6;
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

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  0, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  1, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  1, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  2, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  2, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  3, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  3, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  4, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  4, 1> { enum { index = 4 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  5, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  5, 1> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  6, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  6, 1> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  7, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  7, 1> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  8, 0> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  8, 1> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  9, 0> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology,  9, 1> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology, 10, 0> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology, 10, 1> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology, 11, 0> { enum { index = 7 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshEdgeTopology, 11, 1> { enum { index = 4 }; };


template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 0, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 0, 2> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 0, 3> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 1, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 1, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 1, 2> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 1, 3> { enum { index = 4 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 2, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 2, 1> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 2, 2> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 2, 3> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 3, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 3, 1> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 3, 2> { enum { index = 7 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 3, 3> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 4, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 4, 1> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 4, 2> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 4, 3> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 5, 0> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 5, 1> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 5, 2> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTopology, tnlMeshQuadrilateralTopology, 5, 3> { enum { index = 7 }; };

#endif /* TNLMESHHEXAHEDRONTOPOLOGY_H_ */
