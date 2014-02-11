/***************************************************************************
                          tnlMeshTriangleTag.h  -  description
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

#ifndef TNLMESHTRIANGLETAG_H_
#define TNLMESHTRIANGLETAG_H_

#include <mesh/topology/tnlMeshEdgeTag.h>

struct tnlMeshTriangleTag
{
   enum { dimension = 2 };
};


template<>
struct tnlSubentities< tnlMeshTriangleTag, 0 >
{
   typedef tnlMeshVertexTag Tag;

   enum { count = 3 };
};

template<>
struct tnlSubentities< tnlMeshTriangleTag, 1>
{
   typedef tnlMeshEdgeTag Tag;

   enum { count = 3 };
};


template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTriangleTag, tnlMeshEdgeTag, 2, 1> { enum { index = 1 }; };


#endif /* TNLMESHTRIANGLETAG_H_ */
