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
namespace Topologies {

struct Triangle
{
   static constexpr int dimension = 2;

   static String getType()
   {
      return "MeshTriangleTopology";
   }
};


template<>
struct Subtopology< Triangle, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 3;
};

template<>
struct Subtopology< Triangle, 1 >
{
   typedef Edge Topology;

   static constexpr int count = 3;
};


template<> struct SubentityVertexMap< Triangle, Edge, 0, 0> { enum { index = 1 }; };
template<> struct SubentityVertexMap< Triangle, Edge, 0, 1> { enum { index = 2 }; };

template<> struct SubentityVertexMap< Triangle, Edge, 1, 0> { enum { index = 2 }; };
template<> struct SubentityVertexMap< Triangle, Edge, 1, 1> { enum { index = 0 }; };

template<> struct SubentityVertexMap< Triangle, Edge, 2, 0> { enum { index = 0 }; };
template<> struct SubentityVertexMap< Triangle, Edge, 2, 1> { enum { index = 1 }; };

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
