/***************************************************************************
                          tnlMeshSubentitiesTraits.h  -  description
                             -------------------
    begin                : Feb 12, 2014
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

#ifndef TNLMESHSUBENTITIESTRAITS_H_
#define TNLMESHSUBENTITIESTRAITS_H_

#include <core/arrays/tnlStaticArray.h>
#include <core/arrays/tnlSharedArray.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSubentitiesTraits
{
   enum { storageEnabled = tnlMeshSubentityStorage< ConfigTag,
                                                    EntityTag,
                                                    DimensionsTraits::value >::enabled };

   typedef typename ConfigTag::GlobalIndexType                  GlobalIndexType;
   typedef typename ConfigTag::LocalIndexType                   LocalIndexType;
   typedef tnlSubentities< EntityTag, DimensionsTraits::value > Tag;

public:
   typedef tnlMeshEntity< ConfigTag, EntityTag >                 EntityType;
   typedef typename Tag::Tag                                     SubentityTag;
   typedef tnlMeshEntity< ConfigTag, SubentityTag >              SubentityType;

   typedef tnlStorageTraits< storageEnabled >                    SubentityStorageTag;

   enum { count = Tag::count };

   typedef tnlStaticArray< count, GlobalIndexType >              ContainerType;
   typedef tnlSharedArray< GlobalIndexType,
                           tnlHost,
                           LocalIndexType >                      SharedContainerType;
   typedef tnlStaticArray< count, SubentityType >                SubentityContainerType;

   template< LocalIndexType subentityIndex,
             LocalIndexType subentityVertexIndex >
   struct Vertex
   {
      enum { index = tnlSubentityVertex< EntityTag,
                                         SubentityTag,
                                         subentityIndex,
                                         subentityVertexIndex>::index };
   };
};



#endif /* TNLMESHSUBENTITIESTRAITS_H_ */
