/***************************************************************************
                          tnlMeshTraits.h  -  description
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

#ifndef TNLMESHTRAITS_H_
#define TNLMESHTRAITS_H_

#include <core/vectors/tnlStaticVector.h>
#include <core/arrays/tnlArray.h>
#include <mesh/tnlDimensionsTag.h>


template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntitySeed;


template< typename MeshConfig,
          typename Device = tnlHost >
class tnlMeshTraits
{
   public:
      
      static const int meshDimensions = MeshConfig::CellTopology::dimensions;
      static const int worldDimensions = MeshConfig::worldDimensions;
      
      typedef tnlDimensionsTag< meshDimensions >                                   DimensionsTag;

      typedef Device                                                               DeviceType;
      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;      
      
      typedef tnlStaticVector< worldDimensions, typename MeshConfig::RealType >    PointType;
      typedef typename MeshConfig::CellTopology                                    CellTopology;
      typedef tnlMeshEntity< MeshConfig, CellTopology >                            CellEntity;
      typedef tnlMeshEntitySeed< MeshConfig, CellTopology >                        CellSeedType;
      
      typedef tnlArray< PointType, tnlHost, GlobalIndexType >                      PointArrayType;
      typedef tnlArray< CellSeedType, tnlHost, GlobalIndexType >                   CellSeedArrayType;
};


#endif /* TNLMESHTRAITS_H_ */
