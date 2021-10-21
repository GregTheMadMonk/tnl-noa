/***************************************************************************
                          Traits.h  -  description
                             -------------------
    begin                : Nov 9, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool sensible = (DimensionTag::value <= MeshConfig::meshDimension) >
struct WeakStorageTrait
{
   static constexpr bool entityTagsEnabled = MeshConfig::entityTagsStorage( DimensionTag::value );
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
struct WeakStorageTrait< MeshConfig, Device, DimensionTag, false >
{
   static constexpr bool entityTagsEnabled = false;
};

// Entity tags are used in a bitset fashion. Unused bits are available for
// user needs, but these bits should not be changed by users.
enum EntityTags : std::uint8_t
{
   BoundaryEntity = 1,
   GhostEntity = 2,
};

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
