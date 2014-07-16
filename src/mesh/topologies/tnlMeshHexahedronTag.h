/***************************************************************************
                          tnlMeshHexahedronTag.h  -  description
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

#ifndef TNLMESHHEXAHEDRONTAG_H_
#define TNLMESHHEXAHEDRONTAG_H_

#include <mesh/topologies/tnlMeshQuadrilateralTag.h>

struct tnlMeshHexahedronTag
{
   enum { dimensions = 3 };
};

template<>
struct tnlSubentities< tnlMeshHexahedronTag, 0 >
{
   typedef tnlMeshVertexTag Tag;

   enum { count = 8 };
};

template<>
struct tnlSubentities< tnlMeshHexahedronTag, 1 >
{
   typedef tnlMeshEdgeTag Tag;

   enum { count = 12 };
};

template<>
struct tnlSubentities< tnlMeshHexahedronTag, 2 >
{
   typedef tnlMeshQuadrilateralTag Tag;

   enum { count = 6 };
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

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  0, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  1, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  1, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  2, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  2, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  3, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  3, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  4, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  4, 1> { enum { index = 4 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  5, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  5, 1> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  6, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  6, 1> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  7, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  7, 1> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  8, 0> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  8, 1> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  9, 0> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag,  9, 1> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag, 10, 0> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag, 10, 1> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag, 11, 0> { enum { index = 7 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshEdgeTag, 11, 1> { enum { index = 4 }; };


template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 0, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 0, 2> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 0, 3> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 1, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 1, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 1, 2> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 1, 3> { enum { index = 4 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 2, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 2, 1> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 2, 2> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 2, 3> { enum { index = 5 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 3, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 3, 1> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 3, 2> { enum { index = 7 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 3, 3> { enum { index = 6 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 4, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 4, 1> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 4, 2> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 4, 3> { enum { index = 7 }; };

template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 5, 0> { enum { index = 4 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 5, 1> { enum { index = 5 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 5, 2> { enum { index = 6 }; };
template<> struct tnlSubentityVertex< tnlMeshHexahedronTag, tnlMeshQuadrilateralTag, 5, 3> { enum { index = 7 }; };

#endif /* TNLMESHHEXAHEDRONTAG_H_ */
