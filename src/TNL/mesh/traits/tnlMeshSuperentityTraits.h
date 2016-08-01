/***************************************************************************
                          tnlMeshSuperentityTraits.h  -  description
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
#include <TNL/mesh/tnlMeshEntity.h>
#include <TNL/mesh/config/tnlMeshConfigBase.h>
#include <TNL/mesh/topologies/tnlMeshEntityTopology.h>
#include <TNL/mesh/traits/tnlMeshEntityTraits.h>
#include <TNL/core/multimaps/tnlEllpackIndexMultimap.h>
#include <TNL/mesh/layers/tnlMeshSuperentityAccessor.h>

namespace TNL {

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class tnlMeshSuperentityTraits
{
   public:
 
   typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshConfig::LocalIndexType                               LocalIndexType;


   static const bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimensions );
   //typedef tnlStorageTraits< storageEnabled >                               SuperentityStorageTag;
   typedef tnlMeshEntity< MeshConfig, EntityTopology >                            EntityType;
   typedef tnlMeshEntityTraits< MeshConfig, Dimensions >                     EntityTraits;
   typedef typename EntityTraits::EntityTopology                             SuperentityTopology;
   typedef typename EntityTraits::EntityType                                 SuperentityType;


   /****
    * Type of container for storing of the superentities indecis.
    */
   typedef Arrays::Array< GlobalIndexType, Devices::Host, LocalIndexType >             StorageArrayType;
 
   typedef tnlEllpackIndexMultimap< GlobalIndexType, Devices::Host >                        StorageNetworkType;
   typedef tnlMeshSuperentityAccessor< typename StorageNetworkType::ValuesAccessorType > SuperentityAccessorType;
 
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

} // namespace TNL
