/***************************************************************************
                          MeshSuperentityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Arrays/Array.h>
#include <TNL/Arrays/ConstSharedArray.h>
#include <TNL/List.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/core/multimaps/tnlEllpackIndexMultimap.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityAccessor.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class MeshSuperentityTraits
{
   public:
 
   typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshConfig::LocalIndexType                               LocalIndexType;


   static const bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimensions );
   //typedef tnlStorageTraits< storageEnabled >                               SuperentityStorageTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                            EntityType;
   typedef MeshEntityTraits< MeshConfig, Dimensions >                     EntityTraits;
   typedef typename EntityTraits::EntityTopology                             SuperentityTopology;
   typedef typename EntityTraits::EntityType                                 SuperentityType;


   /****
    * Type of container for storing of the superentities indecis.
    */
   typedef Arrays::Array< GlobalIndexType, Devices::Host, LocalIndexType >             StorageArrayType;
 
   typedef tnlEllpackIndexMultimap< GlobalIndexType, Devices::Host >                        StorageNetworkType;
   typedef MeshSuperentityAccessor< typename StorageNetworkType::ValuesAccessorType > SuperentityAccessorType;
 
   /****
    * Type for passing the superentities indecis by the getSuperentitiesIndices()
    * method. We introduce it because of the compatibility with the subentities
    * which are usually stored in static array.
    */
   typedef Arrays::SharedArray< GlobalIndexType, Devices::Host, LocalIndexType >       AccessArrayType;

   /****
    * This is used by the mesh initializer.
    */
   typedef List< GlobalIndexType >                                       GrowableContainerType;

};

} // namespace Meshes
} // namespace TNL
