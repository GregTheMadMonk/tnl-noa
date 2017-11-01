/***************************************************************************
                          MeshTraits.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/Topologies/Vertex.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeed;
template< typename MeshConfig, typename Device, int Dimension > class MeshEntityTraits;
template< typename MeshConfig, typename Device, typename MeshEntity, int Subdimension > class MeshSubentityTraits;
template< typename MeshConfig, typename Device, typename MeshEntity, int Superdimension > class MeshSuperentityTraits;

template< typename MeshConfig,
          typename Device = Devices::Host >
class MeshTraits
{
public:
   static constexpr int meshDimension  = MeshConfig::CellTopology::dimension;
   static constexpr int worldDimension = MeshConfig::worldDimension;

   using DeviceType        = Device;
   using GlobalIndexType   = typename MeshConfig::GlobalIndexType;
   using LocalIndexType    = typename MeshConfig::LocalIndexType;

   using CellTopology      = typename MeshConfig::CellTopology;
   using CellType          = MeshEntity< MeshConfig, Device, CellTopology >;
   using VertexType        = MeshEntity< MeshConfig, Device, Topologies::Vertex >;
   using PointType         = Containers::StaticVector< worldDimension, typename MeshConfig::RealType >;
   using CellSeedType      = MeshEntitySeed< MeshConfig, CellTopology >;

   using PointArrayType    = Containers::Array< PointType, DeviceType, GlobalIndexType >;
   using CellSeedArrayType = Containers::Array< CellSeedType, DeviceType, GlobalIndexType >;
   using BoundaryTagsArrayType = Containers::Array< bool, DeviceType, GlobalIndexType >;
   using GlobalIndexOrderingArrayType = Containers::Array< GlobalIndexType, DeviceType, GlobalIndexType >;

   template< int Dimension >
   using EntityTraits = MeshEntityTraits< MeshConfig, DeviceType, Dimension >;

   template< typename EntityTopology, int Subdimension >
   using SubentityTraits = MeshSubentityTraits< MeshConfig, DeviceType, EntityTopology, Subdimension >;

   template< typename EntityTopology, int Superdimension >
   using SuperentityTraits = MeshSuperentityTraits< MeshConfig, DeviceType, EntityTopology, Superdimension >;

   using DimensionTag = Meshes::DimensionTag< meshDimension >;
};

} // namespace Meshes
} // namespace TNL
