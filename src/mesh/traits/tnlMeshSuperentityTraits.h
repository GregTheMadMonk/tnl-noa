/***************************************************************************
                          tnlMeshSuperentityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHSUPERENTITYTRAITS_H_
#define TNLMESHSUPERENTITYTRAITS_H_

#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlConstSharedArray.h>
#include <core/tnlList.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/traits/tnlMeshEntityTraits.h>
#include <core/multimaps/tnlEllpackIndexMultimap.h>
#include <mesh/layers/tnlMeshSuperentityAccessor.h>

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
   typedef tnlArray< GlobalIndexType, tnlHost, LocalIndexType >             StorageArrayType;
   
   typedef tnlEllpackIndexMultimap< GlobalIndexType, tnlHost >                        StorageNetworkType;
   typedef tnlMeshSuperentityAccessor< typename StorageNetworkType::ValuesAccessorType > SuperentityAccessorType;
   
   /****
    * Type for passing the superentities indecis by the getSuperentitiesIndices()
    * method. We introduce it because of the compatibility with the subentities
    * which are usually stored in static array.
    */
   typedef tnlSharedArray< GlobalIndexType, tnlHost, LocalIndexType >       AccessArrayType;

   /****
    * This is used by the mesh initializer.
    */
   typedef tnlList< GlobalIndexType >                                       GrowableContainerType;

};


#endif /* TNLMESHSUPERENTITYTRAITS_H_ */
