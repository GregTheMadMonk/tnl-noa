/***************************************************************************
                          MeshSuperentityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Containers/Multimaps/EllpackIndexMultimap.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          int Dimension >
class MeshSuperentityTraits
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   static_assert( EntityTopology::dimension < Dimension, "Superentity dimension must be higher than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimension );

   using SuperentityTopology = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityTopology;
   using SuperentityType     = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityType;

   /****
    * Type of container for storing of the superentities indices.
    */
   // TODO: write general operator= for different SliceSize and remove the '32' here
   using StorageNetworkType      = Containers::Multimaps::EllpackIndexMultimap< GlobalIndexType, Device, LocalIndexType, 32 >;
};

} // namespace Meshes
} // namespace TNL
