/***************************************************************************
                          tnlMeshSuperentitiesTraits.h  -  description
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

#ifndef TNLMESHSUPERENTITIESTRAITS_H_
#define TNLMESHSUPERENTITIESTRAITS_H_

#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlConstSharedArray.h>
#include <core/tnlList.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/traits/tnlMeshEntitiesTraits.h>

template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentitiesTraits
{
   typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
   typedef typename MeshConfig::LocalIndexType                               LocalIndexType;

   public:

   static const bool storageEnabled = MeshConfig::template superentityStorage< EntityTag >( EntityTag(), DimensionsTag::value );
   typedef tnlMeshEntity< MeshConfig, EntityTag >                            EntityType;
   typedef typename
      tnlMeshEntitiesTraits< MeshConfig,
                             DimensionsTag::value >::Tag                        SuperentityTag;
   typedef typename
      tnlMeshEntitiesTraits< MeshConfig,
                             DimensionsTag::value >::EntityType                       SuperentityType;

   typedef tnlStorageTraits< storageEnabled >                               SuperentityStorageTag;

   /****
    * Type of container for storing of the superentities indecis.
    */
   typedef tnlArray< GlobalIndexType, tnlHost, LocalIndexType >             ContainerType;

   /****
    * Type for passing the superentities indecis by the getSuperentitiesIndices()
    * method. We introduce it because of the compatibility with the subentities
    * which are usually stored in static array.
    */
   typedef tnlSharedArray< GlobalIndexType, tnlHost, LocalIndexType >       SharedContainerType;

   /****
    * This is used by the mesh initializer.
    */
   typedef tnlList< GlobalIndexType >                                       GrowableContainerType;

};


#endif /* TNLMESHSUPERENTITIESTRAITS_H_ */
