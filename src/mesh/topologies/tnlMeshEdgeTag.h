/***************************************************************************
                          tnlMeshEdgeTag.h  -  description
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

#ifndef TNLMESHEDGETAG_H_
#define TNLMESHEDGETAG_H_

#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/topologies/tnlMeshVertexTag.h>

struct tnlMeshEdgeTag
{
   enum { dimensions = 1 };
};


template<>
struct tnlSubentities< tnlMeshEdgeTag, 0 >
{
   typedef tnlMeshVertexTag Tag;

   enum { count = 2 };
};

#endif /* TNLMESHEDGETAG_H_ */