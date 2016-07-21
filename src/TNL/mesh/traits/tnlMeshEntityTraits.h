/***************************************************************************
                          tnlMeshEntityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Vectors/StaticVector.h>
#include <TNL/Arrays/Array.h>
#include <TNL/Arrays/SharedArray.h>
#include <TNL/Arrays/ConstSharedArray.h>
#include <TNL/core/tnlIndexedSet.h>
#include <TNL/mesh/topologies/tnlMeshEntityTopology.h>
#include <TNL/mesh/config/tnlMeshConfigBase.h>
#include <TNL/mesh/traits/tnlMeshTraits.h>

namespace TNL {

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

	typedef typename tnlMeshTraits< MeshConfig >::template EntityTraits< SuperDimensionsTag::value >::EntityTopology SuperentityTopology;

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

      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;
      typedef typename tnlMeshEntityTopology< MeshConfig, Dimensions >::Topology   EntityTopology;
 
      typedef tnlMeshEntity< MeshConfig, EntityTopology >                          EntityType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >                      SeedType;
      typedef tnlMeshEntityReferenceOrientation< MeshConfig, EntityTopology >      ReferenceOrientationType;
      typedef tnlMeshEntitySeedKey< MeshConfig, EntityTopology >                   Key;


      typedef Arrays::Array< EntityType, tnlHost, GlobalIndexType >               StorageArrayType;
      typedef Arrays::tnlSharedArray< EntityType, tnlHost, GlobalIndexType >         AccessArrayType;
      typedef tnlIndexedSet< EntityType, GlobalIndexType, Key >                      UniqueContainerType;
      typedef tnlIndexedSet< SeedType, GlobalIndexType, Key >                        SeedIndexedSetType;
      typedef Arrays::Array< SeedType, tnlHost, GlobalIndexType >                 SeedArrayType;
      typedef Arrays::Array< ReferenceOrientationType, tnlHost, GlobalIndexType > ReferenceOrientationArrayType;

      typedef Arrays::tnlConstSharedArray< EntityType, tnlHost, GlobalIndexType >    SharedArrayType;
};

} // namespace TNL
