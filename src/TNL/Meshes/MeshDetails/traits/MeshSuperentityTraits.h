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

#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Experimental/Multimaps/EllpackIndexMultimap.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          int Dimension >
class MeshSuperentityTraits
{
public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   // FIXME: this would break MeshSuperentityAccess, but it should be possible to implement it similarly to MeshSubentityAccess
   //static_assert( EntityTopology::dimension < Dimension, "Superentity dimension must be higher than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimension );

   using GlobalIndexType     = typename MeshConfig::GlobalIndexType;
   using LocalIndexType      = typename MeshConfig::LocalIndexType;
   using EntityType          = MeshEntity< MeshConfig, EntityTopology >;
   using SuperentityTopology = typename MeshEntityTraits< MeshConfig, Dimension >::EntityTopology;
   using SuperentityType     = typename MeshEntityTraits< MeshConfig, Dimension >::EntityType;

   /****
    * Type of container for storing of the superentities indices.
    */
   using StorageNetworkType      = EllpackIndexMultimap< GlobalIndexType, Devices::Host, LocalIndexType >;
   using SuperentityAccessorType = typename StorageNetworkType::ValuesAccessorType;
};

} // namespace Meshes
} // namespace TNL
