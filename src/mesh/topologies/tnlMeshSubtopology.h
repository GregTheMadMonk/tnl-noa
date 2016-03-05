/***************************************************************************
                          tnlMeshSubtopology.h  -  description
                             -------------------
    begin                : Aug 29, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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

#ifndef TNLMESHSUBTOPOLOGY_H
#define	TNLMESHSUBTOPOLOGY_H

template< typename Topology,
          int dimensions >
class tnlMeshSubtopology;

template< typename Topology,
          typename Subtopology,
          int subtopologyIndex,
          int vertexIndex >
struct tnlMeshSubtopologyVertex;

#endif	/* TNLMESHSUBTOPOLOGY_H */

