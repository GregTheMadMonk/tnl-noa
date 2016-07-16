/***************************************************************************
                          tnlMeshEdgeTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
