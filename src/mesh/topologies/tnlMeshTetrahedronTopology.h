/***************************************************************************
                          tnlMeshTetrahedronTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/topologies/tnlMeshTriangleTopology.h>

namespace TNL {

struct tnlMeshTetrahedronTopology
{
   static const int dimensions = 3;
};

template<>
struct tnlMeshSubtopology< tnlMeshTetrahedronTopology, 0 >
{
   typedef tnlMeshVertexTopology Topology;

   static const int count = 4;
};

template<>
struct tnlMeshSubtopology< tnlMeshTetrahedronTopology, 1 >
{
   typedef tnlMeshEdgeTopology Topology;

   static const int count = 6;
};

template<>
struct tnlMeshSubtopology< tnlMeshTetrahedronTopology, 2 >
{
   typedef tnlMeshTriangleTopology Topology;

   static const int count = 4;
};


template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 2, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 3, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 4, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshEdgeTopology, 5, 1> { enum { index = 3 }; };


template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 0, 2> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 1, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 1> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 2, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 1> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTopology, tnlMeshTriangleTopology, 3, 2> { enum { index = 3 }; };

} // namespace TNL
