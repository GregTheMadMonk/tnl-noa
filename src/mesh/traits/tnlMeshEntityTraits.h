/***************************************************************************
                          tnlMeshEntityTraits.h  -  description
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

#ifndef TNLMESHENTITYTRAITS_H_
#define TNLMESHENTITYTRAITS_H_

#include <core/vectors/tnlStaticVector.h>
#include <core/arrays/tnlArray.h>
#include <core/arrays/tnlSharedArray.h>
#include <core/arrays/tnlConstSharedArray.h>
#include <core/tnlIndexedSet.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/traits/tnlMeshTraits.h>

template< typename MeshConfig, typename EntityTopology > class tnlMeshEntity;
template< typename MeshConfig, typename EntityTopology > class tnlMeshEntitySeed;
template< typename MeshConfig, typename EntityTopology > class tnlMeshEntitySeedKey;
template< typename MeshConfig, typename EntityTopology > class tnlMeshEntityReferenceOrientation;

template< typename MeshConfig,
          typename DimensionsTag,
          typename SuperDimensionsTag = tnlDimensionsTag< MeshConfig::meshDimensions > >
class tnlMeshEntityOrientationNeeded
{
	static_assert( 0 <= DimensionsTag::value && DimensionsTag::value < MeshConfig::CellTopology::dimensions, "invalid dimensions" );
	static_assert( DimensionsTag::value < SuperDimensionsTag::value && SuperDimensionsTag::value <= MeshConfig::CellTopology::dimensions, "invalid superentity dimensions");

	typedef typename tnlMeshTraits< MeshConfig >::template EntityTraits< SuperDimensionsTag::value >::Tag SuperentityTopology;

	static const bool previousSuperDimensionsValue = tnlMeshEntityOrientationNeeded< MeshConfig, DimensionsTag, typename SuperDimensionsTag::Decrement >::value;
	static const bool thisSuperDimensionsValue = tnlMeshTraits< MeshConfig >::template SubentityTraits< SuperentityTopology, DimensionsTag::value >::orientationEnabled;

   public:
      static const bool value = ( previousSuperDimensionsValue || thisSuperDimensionsValue );
};

template< typename MeshConfig, typename DimensionsTag >
class tnlMeshEntityOrientationNeeded< MeshConfig, DimensionsTag, DimensionsTag >
{
	static_assert( 0 <= DimensionsTag::value && DimensionsTag::value <= MeshConfig::CellTopology::dimensions, "invalid dimensions" );

   public:
      static const bool value = false;
};


template< typename MeshConfig,
          int Dimensions >
class tnlMeshEntityTraits
{   
   public:

      static const bool storageEnabled = MeshConfig::entityStorage( Dimensions );
      static const bool orientationNeeded = tnlMeshEntityOrientationNeeded< MeshConfig, tnlDimensionsTag< Dimensions > >::value;
      //typedef tnlStorageTraits< storageEnabled >                     EntityStorageTag;

      typedef typename MeshConfig::GlobalIndexType                              GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                               LocalIndexType;
      typedef typename tnlMeshEntityTopology< MeshConfig,
                                           tnlDimensionsTag< Dimensions > >::Tag      EntityTag;
      
      typedef EntityTag                                              Tag;
      typedef tnlMeshEntity< MeshConfig, Tag >                        EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTag >              SeedType;
      typedef tnlMeshEntityReferenceOrientation< MeshConfig, EntityTag >        ReferenceOrientationType;
      typedef tnlMeshEntitySeedKey< MeshConfig, EntityTag >               Key;


      typedef tnlArray< EntityType, tnlHost, GlobalIndexType >                  StorageArrayType;
      typedef tnlSharedArray< EntityType, tnlHost, GlobalIndexType >            AccessArrayType;
      typedef tnlIndexedSet< EntityType, GlobalIndexType, Key >            UniqueContainerType;
      typedef tnlIndexedSet< SeedType, GlobalIndexType, Key >        SeedIndexedSetType;
      typedef tnlArray< SeedType, tnlHost, GlobalIndexType >         SeedArrayType;
      typedef tnlArray< ReferenceOrientationType, tnlHost, GlobalIndexType > ReferenceOrientationArrayType;

      typedef tnlConstSharedArray< EntityType, tnlHost, GlobalIndexType >  SharedArrayType;
};


#endif /* TNLMESHENTITYTRAITS_H_ */
