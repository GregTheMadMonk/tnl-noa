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
#include <mesh/tnlDimensionsTag.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity;

template< typename MeshConfig,
          typename Device = tnlHost >
class tnlMeshTraits
{
   public:
      
      static const int meshDimensions = MeshConfig::CellType::dimensions;
      static const int worldDimensions = MeshConfig::worldDimensions;

      typedef Device                                                               DeviceType;
      typedef typename MeshConfig::GlobalIndexTyp                                  GlobalIndexType;
	   typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;      
      
      typedef tnlStaticVector< worldDimensions, typename MeshConfig::RealType >    PointType;
      typedef tnlMeshEntity< MeshConfig, typename MeshConfig::CellType >           CellType;
      typedef typename CellType::SeedType                                          CellSeedType;
      
      typedef tnlArray< PointType, tnlHost, GlobalIndexType >                      PointArrayType;
	   typedef tnlArray< CellSeedType, tnlHost, GlobalIndexType >                   CellSeedArrayType;
};


#endif /* TNLMESHTRAITS_H_ */
