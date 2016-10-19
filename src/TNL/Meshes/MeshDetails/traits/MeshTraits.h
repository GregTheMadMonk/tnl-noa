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
   static constexpr int meshDimensions  = MeshConfig::CellTopology::dimensions;
   static constexpr int worldDimensions = MeshConfig::worldDimensions;

   using DeviceType                     = Device;
   using GlobalIndexType                = typename MeshConfig::GlobalIndexType;
   using LocalIndexType                 = typename MeshConfig::LocalIndexType;

   using CellTopology                   = typename MeshConfig::CellTopology;
   using CellType                       = MeshEntity< MeshConfig, CellTopology >;
   using VertexType                     = MeshEntity< MeshConfig, MeshVertexTopology >;
   using PointType                      = Containers::StaticVector< worldDimensions, typename MeshConfig::RealType >;
   using CellSeedType                   = MeshEntitySeed< MeshConfig, CellTopology >;

   using PointArrayType                 = Containers::Array< PointType, Devices::Host, GlobalIndexType >;
   using CellSeedArrayType              = Containers::Array< CellSeedType, Devices::Host, GlobalIndexType >;
   using GlobalIdArrayType              = Containers::Array< GlobalIndexType, Devices::Host, GlobalIndexType >;
   using IdArrayAccessorType            = Containers::tnlConstSharedArray< GlobalIndexType, Devices::Host, LocalIndexType >;
   using IdPermutationArrayAccessorType = Containers::tnlConstSharedArray< LocalIndexType, Devices::Host, LocalIndexType >;

   template< int Dimensions >
   using EntityTraits = MeshEntityTraits< MeshConfig, Dimensions >;

   template< typename EntityTopology, int SubDimensions >
   using SubentityTraits = MeshSubentityTraits< MeshConfig, EntityTopology, SubDimensions >;

   template< typename EntityTopology, int SuperDimensions >
   using SuperentityTraits = MeshSuperentityTraits< MeshConfig, EntityTopology, SuperDimensions >;

   using DimensionsTag = MeshDimensionsTag< meshDimensions >;
};

} // namespace Meshes
} // namespace TNL
