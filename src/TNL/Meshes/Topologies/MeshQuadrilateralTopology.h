/***************************************************************************
                          MeshQuadrilateralTopology.h  -  description
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

struct MeshQuadrilateralTopology
{
   static constexpr int dimension = 2;

   static String getType()
   {
      return "MeshQuadrilateralTopology";
   }
};


template<>
struct MeshSubtopology< MeshQuadrilateralTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static constexpr int count = 4;
};

template<>
struct MeshSubtopology< MeshQuadrilateralTopology, 1 >
{
   typedef MeshEdgeTopology Topology;

   static constexpr int count = 4;
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

template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 0, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 0, 1> { enum { index = 1 }; };

template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 1, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 1, 1> { enum { index = 2 }; };

template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 2, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 2, 1> { enum { index = 3 }; };

template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 3, 0> { enum { index = 3 }; };
template<> struct SubentityVertexMap< MeshQuadrilateralTopology, MeshEdgeTopology, 3, 1> { enum { index = 0 }; };

} // namespace Meshes
} // namespace TNL
