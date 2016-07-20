/***************************************************************************
                          tnlMeshTraits.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Vectors/StaticVector.h>
#include <TNL/Arrays/Array.h>
#include <TNL/Arrays/SharedArray.h>
#include <TNL/Arrays/ConstSharedArray.h>
#include <TNL/mesh/tnlDimensionsTag.h>

namespace TNL {

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
 
      typedef Arrays::Array< PointType, tnlHost, GlobalIndexType >                  PointArrayType;
      typedef Arrays::Array< CellSeedType, tnlHost, GlobalIndexType >               CellSeedArrayType;
      typedef Arrays::Array< GlobalIndexType, tnlHost, GlobalIndexType >            GlobalIdArrayType;
      typedef Arrays::tnlConstSharedArray< GlobalIndexType, tnlHost, LocalIndexType >  IdArrayAccessorType;
      typedef Arrays::tnlConstSharedArray< LocalIndexType, tnlHost, LocalIndexType >   IdPermutationArrayAccessorType;
 
      template< int Dimensions > using EntityTraits =
         tnlMeshEntityTraits< MeshConfig, Dimensions >;
 
      template< typename EntityTopology, int SubDimensions > using SubentityTraits =
         tnlMeshSubentityTraits< MeshConfig, EntityTopology, SubDimensions >;
 
      template< typename EntityTopology, int SuperDimensions > using SuperentityTraits =
         tnlMeshSuperentityTraits< MeshConfig, EntityTopology, SuperDimensions >;
 
 
      typedef tnlDimensionsTag< meshDimensions >                                   DimensionsTag;

};

} // namespace TNL
