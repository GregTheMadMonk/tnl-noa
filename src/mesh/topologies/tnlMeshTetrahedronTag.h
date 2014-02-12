/***************************************************************************
                          tnlMeshTetrahedronTag.h  -  description
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

#ifndef TNLMESHTETRAHEDRONTAG_H_
#define TNLMESHTETRAHEDRONTAG_H_

#include <mesh/topologies/tnlMeshTriangleTag.h>

struct tnlMeshTetrahedronTag
{
   enum { dimensions = 3 };
};

template<>
struct tnlSubentities< tnlMeshTetrahedronTag, 0 >
{
   typedef tnlMeshVertexTag Tag;

   enum { count = 4 };
};

template<>
struct tnlSubentities< tnlMeshTetrahedronTag, 1 >
{
   typedef tnlMeshEdgeTag Tag;

   enum { count = 6 };
};

template<>
struct tnlSubentities< tnlMeshTetrahedronTag, 2 >
{
   typedef tnlMeshTriangleTag Tag;

   enum { count = 4 };
};


template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 0, 1> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 1, 1> { enum { index = 0 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 2, 1> { enum { index = 1 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 3, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 4, 1> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshEdgeTag, 5, 1> { enum { index = 3 }; };


template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 0, 2> { enum { index = 2 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 0> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 1> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 1, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 0> { enum { index = 1 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 1> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 2, 2> { enum { index = 3 }; };

template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 0> { enum { index = 2 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 1> { enum { index = 0 }; };
template<> struct tnlSubentityVertex< tnlMeshTetrahedronTag, tnlMeshTriangleTag, 3, 2> { enum { index = 3 }; };


#endif /* TNLMESHTETRAHEDRONTAG_H_ */
