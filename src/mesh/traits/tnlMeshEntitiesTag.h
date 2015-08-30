/***************************************************************************
                          tnlMeshEntitiesTag.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#ifndef TNLMESHENTITIESTAG_H_
#define TNLMESHENTITIESTAG_H_

#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>

template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshEntitiesTag
{
   public:

   typedef typename tnlSubentities< typename MeshConfig::CellTopology,
                                    DimensionsTag::value >::Tag Tag;
};

template< typename MeshConfig >
class tnlMeshEntitiesTag< MeshConfig,
                          tnlDimensionsTag< MeshConfig::CellTopology::dimensions > >
{
   public:

   typedef typename MeshConfig::CellTopology Tag;
};


#endif /* TNLMESHENTITIESTAG_H_ */
