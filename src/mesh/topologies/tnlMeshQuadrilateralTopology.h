/***************************************************************************
                          tnlMeshQuadrilateralTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHQUADRILATERALTOPOLOGY_H_
#define TNLMESHQUADRILATERALTOPOLOGY_H_

#include <mesh/topologies/tnlMeshEdgeTopology.h>

struct tnlMeshQuadrilateralTopology
{
   static const int dimensions = 2;
};


template<>
struct tnlMeshSubtopology< tnlMeshQuadrilateralTopology, 0 >
{
   typedef tnlMeshVertexTopology Topology;

   static const int count = 4;
};

template<>
struct tnlMeshSubtopology< tnlMeshQuadrilateralTopology, 1 >
{
   typedef tnlMeshEdgeTopology Topology;

   static const int count = 4;
};


/****
 * Indexing of the vertices follows the VTK file format
 *
 *   3                     2
 *    +-------------------+
 *    |                   |
 *    |                   |
 *    |                   |
 *    |                   |
 *    |                   |
 *    +-------------------+
 *   0                     1
 *
 * The edges are indexed as follows:
 *
 *              2
 *    +-------------------+
 *    |                   |
 *    |                   |
 *  3 |                   | 1
 *    |                   |
 *    |                   |
 *    +-------------------+
 *              0
 *
 */

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 0, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 1, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 1, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 2, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 2, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 3, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTopology, tnlMeshEdgeTopology, 3, 1> { enum { index = 0 }; };


#endif /* TNLMESHQUADRILATERALTOPOLOGY_H_ */
