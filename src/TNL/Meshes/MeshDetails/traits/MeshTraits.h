/***************************************************************************
                          MeshTraits.h  -  description
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
#include <TNL/Meshes/MeshDimensionsTag.h>

namespace TNL {
namespace Meshes {

struct MeshVertexTopology;
template< typename MeshConfig, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeed;
template< typename MeshConfig, int Dimensions > class MeshEntityTraits;
template< typename MeshConfig, typename MeshEntity, int SubDimensions > class MeshSubentityTraits;
template< typename MeshConfig, typename MeshEntity, int SuperDimensions > class MeshSuperentityTraits;

template< typename MeshConfig,
          typename Device = Devices::Host >
class MeshTraits
{
   public:
 
      static const int meshDimensions = MeshConfig::CellTopology::dimensions;
      static const int worldDimensions = MeshConfig::worldDimensions;

      typedef Device                                                               DeviceType;
      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;
 
      typedef typename MeshConfig::CellTopology                                    CellTopology;
      typedef MeshEntity< MeshConfig, CellTopology >                            CellType;
      typedef MeshEntity< MeshConfig, MeshVertexTopology >                   VertexType;
      typedef Vectors::StaticVector< worldDimensions, typename MeshConfig::RealType >    PointType;
      typedef MeshEntitySeed< MeshConfig, CellTopology >                        CellSeedType;
 
      typedef Arrays::Array< PointType, Devices::Host, GlobalIndexType >                  PointArrayType;
      typedef Arrays::Array< CellSeedType, Devices::Host, GlobalIndexType >               CellSeedArrayType;
      typedef Arrays::Array< GlobalIndexType, Devices::Host, GlobalIndexType >            GlobalIdArrayType;
      typedef Arrays::tnlConstSharedArray< GlobalIndexType, Devices::Host, LocalIndexType >  IdArrayAccessorType;
      typedef Arrays::tnlConstSharedArray< LocalIndexType, Devices::Host, LocalIndexType >   IdPermutationArrayAccessorType;
 
      template< int Dimensions > using EntityTraits =
         MeshEntityTraits< MeshConfig, Dimensions >;
 
      template< typename EntityTopology, int SubDimensions > using SubentityTraits =
         MeshSubentityTraits< MeshConfig, EntityTopology, SubDimensions >;
 
      template< typename EntityTopology, int SuperDimensions > using SuperentityTraits =
         MeshSuperentityTraits< MeshConfig, EntityTopology, SuperDimensions >;
 
 
      typedef MeshDimensionsTag< meshDimensions >                                   DimensionsTag;

};

} // namespace Meshes
} // namespace TNL
