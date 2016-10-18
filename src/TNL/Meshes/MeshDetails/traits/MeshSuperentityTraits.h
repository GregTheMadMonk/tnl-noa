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
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityAccessor.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          int Dimension >
class MeshSuperentityTraits
{
   public:
 
   typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshConfig::LocalIndexType                               LocalIndexType;


   static const bool storageEnabled = MeshConfig::template superentityStorage< EntityTopology >( EntityTopology(), Dimension );
   //typedef tnlStorageTraits< storageEnabled >                               SuperentityStorageTag;
   typedef MeshEntity< MeshConfig, EntityTopology >                            EntityType;
   typedef MeshEntityTraits< MeshConfig, Dimension >                     EntityTraits;
   typedef typename EntityTraits::EntityTopology                             SuperentityTopology;
   typedef typename EntityTraits::EntityType                                 SuperentityType;


   /****
    * Type of container for storing of the superentities indecis.
    */
   typedef Containers::Array< GlobalIndexType, Devices::Host, LocalIndexType >        StorageArrayType;
 
   typedef EllpackIndexMultimap< GlobalIndexType, Devices::Host, LocalIndexType >     StorageNetworkType;
   typedef MeshSuperentityAccessor< typename StorageNetworkType::ValuesAccessorType > SuperentityAccessorType;
 
   /****
    * Type for passing the superentities indices by the getSuperentityIndices()
    * method. We introduce it because of the compatibility with the subentities
    * which are usually stored in static array.
    */
   typedef Containers::SharedArray< GlobalIndexType, Devices::Host, LocalIndexType >       AccessArrayType;

   /****
    * This is used by the mesh initializer.
    */
   typedef Containers::List< GlobalIndexType >                                       GrowableContainerType;

};

} // namespace Meshes
} // namespace TNL
