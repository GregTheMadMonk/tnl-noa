/***************************************************************************
                          Edge.h  -  description
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

#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/Topologies/Vertex.h>

namespace TNL {
namespace Meshes {
namespace Topologies {
   
struct Edge
{
   static constexpr int dimension = 1;

   static String getType()
   {
      return "Topologies::Edge";
   }
};


template<>
struct Subtopology< Edge, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 2;
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
