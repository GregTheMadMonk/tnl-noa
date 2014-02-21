/***************************************************************************
                          tnlMeshEntitiesTraits.h  -  description
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

#ifndef TNLMESHENTITIESTRAITS_H_
#define TNLMESHENTITIESTRAITS_H_

#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlConstSharedArray.h>
#include <mesh/traits/tnlMeshEntitiesTag.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/tnlMeshEntityKey.h>

template< typename ConfigTag,
          typename DimensionsTraits >
class tnlMeshEntitiesTraits
{
   enum { storageEnabled = tnlMeshEntityStorage< ConfigTag,
                                                 DimensionsTraits::value>::enabled };

   typedef typename ConfigTag::GlobalIndexType                    GlobalIndexType;
   typedef typename ConfigTag::LocalIndexType                     LocalIndexType;
   typedef typename tnlMeshEntitiesTag< ConfigTag,
                                        DimensionsTraits >::Tag   EntityTag;
   typedef tnlMeshEntityKey< ConfigTag, EntityTag >               Key;

   public:

   typedef EntityTag                                              Tag;
   typedef tnlMeshEntity< ConfigTag, Tag >                        Type;

   typedef tnlStorageTraits< storageEnabled >                     EntityStorageTag;

   typedef tnlArray< Type, tnlHost, GlobalIndexType >             ContainerType;
   //typedef IndexedSet< Type, GlobalIndexType, Key >             UniqueContainerType;

   typedef tnlConstSharedArray< Type, tnlHost, GlobalIndexType >           SharedArrayType;
};


#endif /* TNLMESHENTITIESTRAITS_H_ */
