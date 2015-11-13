/***************************************************************************
                          tnlMeshTriangleTopology.h  -  description
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

#ifndef TNLMESHTRIANGLETOPOLOGY_H_
#define TNLMESHTRIANGLETOPOLOGY_H_

#include <mesh/topologies/tnlMeshEdgeTopology.h>

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


#endif /* TNLMESHTRIANGLETOPOLOGY_H_ */
