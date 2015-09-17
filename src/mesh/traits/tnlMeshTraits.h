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
#include <core/arrays/tnlSharedArray.h>
#include <core/arrays/tnlConstSharedArray.h>
#include <mesh/tnlDimensionsTag.h>

struct tnlMeshVertexTopology;
template< typename MeshConfig, typename EntityTopology > class tnlMeshEntity;
template< typename MeshConfig, typename EntityTopology > class tnlMeshEntitySeed;
template< typename MeshConfig, int Dimensions > class tnlMeshEntityTraits;
template< typename MeshConfig, typename MeshEntity, int SubDimensions > class tnlMeshSubentityTraits;
template< typename MeshConfig, typename MeshEntity, int SuperDimensions > class tnlMeshSuperentityTraits;

template< typename MeshConfig,
          typename Device = tnlHost >
class tnlMeshTraits
{
   public:
      
      static const int meshDimensions = MeshConfig::CellTopology::dimensions;
      static const int worldDimensions = MeshConfig::worldDimensions;     

      typedef Device                                                               DeviceType;
      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;      
            
      typedef typename MeshConfig::CellTopology                                    CellTopology;
      typedef tnlMeshEntity< MeshConfig, CellTopology >                            CellType;
      typedef tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >                   VertexType;
      typedef tnlStaticVector< worldDimensions, typename MeshConfig::RealType >    PointType;
      typedef tnlMeshEntitySeed< MeshConfig, CellTopology >                        CellSeedType;
      
      typedef tnlArray< PointType, tnlHost, GlobalIndexType >                      PointArrayType;
      typedef tnlArray< CellSeedType, tnlHost, GlobalIndexType >                   CellSeedArrayType;
      typedef tnlArray< GlobalIndexType, tnlHost, GlobalIndexType >                GlobalIdArrayType;
      typedef tnlConstSharedArray< GlobalIndexType, tnlHost, LocalIndexType >      IdArrayAccessorType;
      typedef tnlConstSharedArray< LocalIndexType, tnlHost, LocalIndexType >       IdPermutationArrayAccessorType;
      
      template< int Dimensions > using EntityTraits = 
         tnlMeshEntityTraits< MeshConfig, Dimensions >;
      
      template< typename EntityTopology, int SubDimensions > using SubentityTraits =
         tnlMeshSubentityTraits< MeshConfig, EntityTopology, SubDimensions >;
      
      template< typename EntityTopology, int SuperDimensions > using SuperentityTraits =
         tnlMeshSuperentityTraits< MeshConfig, EntityTopology, SuperDimensions >;
      
      
      typedef tnlDimensionsTag< meshDimensions >                                   DimensionsTag;

};


#endif /* TNLMESHTRAITS_H_ */
