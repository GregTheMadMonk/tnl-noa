/***************************************************************************
                          tnlMeshQuadrilateralTag.h  -  description
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

#ifndef TNLMESHQUADRILATERALTAG_H_
#define TNLMESHQUADRILATERALTAG_H_

#include <mesh/topologies/tnlMeshEdgeTag.h>

struct tnlMeshQuadrilateralTag
{
   enum { dimensions = 2 };
};


template<>
struct tnlSubentities< tnlMeshQuadrilateralTag, 0>
{
   typedef tnlMeshVertexTag Tag;

   enum { count = 4 };
};

template<>
struct tnlSubentities< tnlMeshQuadrilateralTag, 1>
{
   typedef tnlMeshEdgeTag Tag;

   enum { count = 4 };
};


template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 0, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 1, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 1, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 2, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 2, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 3, 0> { enum { index = 3 }; };
template<> struct tnlSubentityVertex< tnlMeshQuadrilateralTag, tnlMeshEdgeTag, 3, 1> { enum { index = 0 }; };


#endif /* TNLMESHQUADRILATERALTAG_H_ */
