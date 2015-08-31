/***************************************************************************
                          tnlMeshTetrahedronTopology.h  -  description
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

#ifndef TNLMESHTETRAHEDRONTOPOLOGY_H_
#define TNLMESHTETRAHEDRONTOPOLOGY_H_

#include <mesh/topologies/tnlMeshTriangleTopology.h>

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


#endif /* TNLMESHTETRAHEDRONTOPOLOGY_H_ */
