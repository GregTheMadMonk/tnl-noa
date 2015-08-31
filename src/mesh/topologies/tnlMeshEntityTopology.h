/***************************************************************************
                          tnlMeshEntityTopology.h  -  description
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

#ifndef TNLMESHENTITYTOPOLOGY_H_
#define TNLMESHENTITYTOPOLOGY_H_

template< typename MeshEntityTopology,
          int SubentityDimension >
struct tnlMeshSubtopology
{
};

template< typename MeshEntityTopology,
          typename SubentityTopology,
          int SubentityIndex,
          int SubentityVertexIndex >
struct tnlSubentityVertex;


template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshEntityTopology
{
   public:

   typedef typename tnlMeshSubtopology< typename MeshConfig::CellTopology,
                                    DimensionsTag::value >::Topology Tag;
};

template< typename MeshConfig >
class tnlMeshEntityTopology< MeshConfig,
                          tnlDimensionsTag< MeshConfig::CellTopology::dimensions > >
{
   public:

   typedef typename MeshConfig::CellTopology Tag;
};
#endif /* TNLMESHENTITYTOPOLOGY_H_ */
