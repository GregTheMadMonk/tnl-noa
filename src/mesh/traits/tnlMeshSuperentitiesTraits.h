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

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSuperentitiesTraits
{
   enum { storageEnabled = tnlMeshSuperentityStorage< ConfigTag,
                                                      EntityTag,
                                                      DimensionsTraits::value >::enabled };

   typedef typename ConfigTag::GlobalIndexType                              GlobalIndexType;
   typedef typename ConfigTag::LocalIndexType                               LocalIndexType;

   public:

   typedef tnlMeshEntity< ConfigTag, EntityTag >                            EntityType;
   typedef typename
      tnlMeshEntitiesTraits< ConfigTag,
                             DimensionsTraits >::Tag                        SuperentityTag;
   typedef typename
      tnlMeshEntitiesTraits< ConfigTag,
                             DimensionsTraits >::Type                       SuperentityType;

   typedef tnlStorageTraits< storageEnabled >                               SuperentityStorageTag;

   typedef tnlArray< GlobalIndexType, tnlHost, LocalIndexType >             ContainerType;
   typedef tnlList< GlobalIndexType >                                       GrowableContainerType;
   typedef tnlConstSharedArray< GlobalIndexType, tnlHost, LocalIndexType >  SharedArrayType;
};


#endif /* TNLMESHSUPERENTITIESTRAITS_H_ */
