/***************************************************************************
                          tnlMeshTriangleTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/topologies/tnlMeshEdgeTopology.h>

namespace TNL {

struct tnlMeshTriangleTopology
{
   static const int dimensions = 2;
};


template<>
struct tnlMeshSubtopology< tnlMeshTriangleTopology, 0 >
{
   typedef tnlMeshVertexTopology Topology;

   static const int count = 3;
};

template<>
struct tnlMeshSubtopology< tnlMeshTriangleTopology, 1 >
{
   typedef tnlMeshEdgeTopology Topology;

   static const int count = 3;
};


template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTopology, tnlMeshEdgeTopology, 2, 1> { enum { index = 1 }; };

} // namespace TNL
