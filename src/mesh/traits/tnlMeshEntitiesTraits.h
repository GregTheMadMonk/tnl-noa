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
#include <core/tnlIndexedSet.h>
#include <mesh/traits/tnlMeshEntitiesTag.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/tnlMeshEntityKey.h>
#include <mesh/tnlMeshEntitySeed.h>

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntitySeed;

template< typename ConfigTag,
          typename DimensionsTag >
class tnlMeshEntitiesTraits
{   
   /*enum { storageEnabled = tnlMeshEntityStorage< ConfigTag,
                                                 DimensionsTag::value>::enabled };*/
   static const bool storageEnabled = ConfigTag::entityStorage( DimensionsTag::value );

   typedef typename ConfigTag::GlobalIndexType                    GlobalIndexType;
   typedef typename ConfigTag::LocalIndexType                     LocalIndexType;
   typedef typename tnlMeshEntitiesTag< ConfigTag,
                                        DimensionsTag >::Tag      EntityTag;
   typedef tnlMeshEntitySeedKey< ConfigTag, EntityTag >               Key;

   public:

   typedef EntityTag                                              Tag;
   typedef tnlMeshEntity< ConfigTag, Tag >                        Type;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >              SeedType;

   typedef tnlStorageTraits< storageEnabled >                     EntityStorageTag;

   typedef tnlArray< Type, tnlHost, GlobalIndexType >             ContainerType;
   typedef tnlSharedArray< Type, tnlHost, GlobalIndexType >       SharedContainerType;
   typedef tnlIndexedSet< Type, GlobalIndexType, Key >            UniqueContainerType;
   typedef tnlIndexedSet< SeedType, GlobalIndexType, Key >        SeedIndexedSetType;
   typedef tnlArray< SeedType, tnlHost, GlobalIndexType >         SeedArrayType;

   typedef tnlConstSharedArray< Type, tnlHost, GlobalIndexType >  SharedArrayType;
};


#endif /* TNLMESHENTITIESTRAITS_H_ */
