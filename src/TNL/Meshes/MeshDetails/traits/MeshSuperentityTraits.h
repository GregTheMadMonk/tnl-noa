/***************************************************************************
                          MeshSuperentityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#include <TNL/Containers/Array.h>
#include <TNL/Containers/ConstSharedArray.h>
#include <TNL/Containers/List.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
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
   static constexpr bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimensions );

   using GlobalIndexType     = typename MeshConfig::GlobalIndexType;
   using LocalIndexType      = typename MeshConfig::LocalIndexType;
   using EntityType          = MeshEntity< MeshConfig, EntityTopology >;
   using EntityTraits        = MeshEntityTraits< MeshConfig, Dimensions >;
   using SuperentityTopology = typename EntityTraits::EntityTopology;
   using SuperentityType     = typename EntityTraits::EntityType;

   /****
    * Type of container for storing of the superentities indices.
    */
   using StorageNetworkType      = EllpackIndexMultimap< GlobalIndexType, Devices::Host, LocalIndexType >;
   using SuperentityAccessorType = typename StorageNetworkType::ValuesAccessorType;
};

} // namespace Meshes
} // namespace TNL
