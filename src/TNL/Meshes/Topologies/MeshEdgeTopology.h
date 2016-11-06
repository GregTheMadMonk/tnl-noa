/***************************************************************************
                          MeshEdgeTopology.h  -  description
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

#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>

namespace TNL {
namespace Meshes {
   
struct MeshEdgeTopology
{
   static constexpr int dimensions = 1;

   static String getType()
   {
      return "MeshEdgeTopology";
   }
};


template<>
struct MeshSubtopology< MeshEdgeTopology, 0 >
{
   typedef MeshVertexTopology Topology;

   static constexpr int count = 2;
};

} // namespace Meshes
} // namespace TNL
