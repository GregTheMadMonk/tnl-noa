/***************************************************************************
                          MeshTraits.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/SharedArray.h>
#include <TNL/Containers/ConstSharedArray.h>
#include <TNL/Meshes/MeshDimensionTag.h>

namespace TNL {
namespace Meshes {

struct MeshVertexTopology;
template< typename MeshConfig, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeed;
template< typename MeshConfig, int Dimension > class MeshEntityTraits;
template< typename MeshConfig, typename MeshEntity, int SubDimension > class MeshSubentityTraits;
template< typename MeshConfig, typename MeshEntity, int SuperDimension > class MeshSuperentityTraits;

template< typename MeshConfig,
          typename Device = Devices::Host >
class MeshTraits
{
   public:
 
      static const int meshDimension = MeshConfig::CellTopology::dimensions;
      static const int worldDimension = MeshConfig::worldDimension;

      typedef Device                                                               DeviceType;
      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;
 
      typedef typename MeshConfig::CellTopology                                    CellTopology;
      typedef MeshEntity< MeshConfig, CellTopology >                            CellType;
      typedef MeshEntity< MeshConfig, MeshVertexTopology >                   VertexType;
      typedef Containers::StaticVector< worldDimension, typename MeshConfig::RealType >    PointType;
      typedef MeshEntitySeed< MeshConfig, CellTopology >                        CellSeedType;
 
      typedef Containers::Array< PointType, Devices::Host, GlobalIndexType >                  PointArrayType;
      typedef Containers::Array< CellSeedType, Devices::Host, GlobalIndexType >               CellSeedArrayType;
      typedef Containers::Array< GlobalIndexType, Devices::Host, GlobalIndexType >            GlobalIdArrayType;
      typedef Containers::tnlConstSharedArray< GlobalIndexType, Devices::Host, LocalIndexType >  IdArrayAccessorType;
      typedef Containers::tnlConstSharedArray< LocalIndexType, Devices::Host, LocalIndexType >   IdPermutationArrayAccessorType;
 
      template< int Dimension > using EntityTraits =
         MeshEntityTraits< MeshConfig, Dimension >;
 
      template< typename EntityTopology, int SubDimension > using SubentityTraits =
         MeshSubentityTraits< MeshConfig, EntityTopology, SubDimension >;
 
      template< typename EntityTopology, int SuperDimension > using SuperentityTraits =
         MeshSuperentityTraits< MeshConfig, EntityTopology, SuperDimension >;
 
 
      typedef MeshDimensionTag< meshDimension >                                   DimensionTag;

};

} // namespace Meshes
} // namespace TNL
