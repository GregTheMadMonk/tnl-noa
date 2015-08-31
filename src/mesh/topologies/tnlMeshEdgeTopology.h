/***************************************************************************
                          tnlMeshEdgeTopology.h  -  description
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

#ifndef TNLMESHEDGETOPOLOGY_H_
#define TNLMESHEDGETOPOLOGY_H_

#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/topologies/tnlMeshVertexTopology.h>

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

#endif /* TNLMESHEDGETOPOLOGY_H_ */
