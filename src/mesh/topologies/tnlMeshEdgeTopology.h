/***************************************************************************
                          tnlMeshEdgeTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/topologies/tnlMeshVertexTopology.h>

namespace TNL {

struct tnlMeshEdgeTopology
{
   static const int dimensions = 1;
};


template<>
struct tnlMeshSubtopology< tnlMeshEdgeTopology, 0 >
{
   typedef tnlMeshVertexTopology Topology;

   static const int count = 2;
};

} // namespace TNL
