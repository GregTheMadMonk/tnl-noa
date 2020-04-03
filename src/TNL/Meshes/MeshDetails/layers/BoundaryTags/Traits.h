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
namespace BoundaryTags {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool sensible = (DimensionTag::value <= MeshConfig::meshDimension) >
struct WeakStorageTrait
{
   using EntityTopology = typename MeshTraits< MeshConfig >::template EntityTraits< DimensionTag::value >::EntityTopology;
   static constexpr bool boundaryTagsEnabled = MeshConfig::boundaryTagsStorage( EntityTopology() );
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
struct WeakStorageTrait< MeshConfig, Device, DimensionTag, false >
{
   static constexpr bool boundaryTagsEnabled = false;
};

} // namespace BoundaryTags
} // namespace Meshes
} // namespace TNL
