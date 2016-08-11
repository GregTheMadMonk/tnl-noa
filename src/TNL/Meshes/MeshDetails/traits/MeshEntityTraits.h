/***************************************************************************
                          MeshEntityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/SharedArray.h>
#include <TNL/Containers/ConstSharedArray.h>
#include <TNL/core/tnlIndexedSet.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology > class MeshEntity;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeed;
template< typename MeshConfig, typename EntityTopology > class MeshEntitySeedKey;
template< typename MeshConfig, typename EntityTopology > class MeshEntityReferenceOrientation;

template< typename MeshConfig,
          typename DimensionsTag,
          typename SuperDimensionsTag = MeshDimensionsTag< MeshConfig::meshDimensions > >
class MeshEntityOrientationNeeded
{
	static_assert( 0 <= DimensionsTag::value && DimensionsTag::value < MeshConfig::CellTopology::dimensions, "invalid dimensions" );
	static_assert( DimensionsTag::value < SuperDimensionsTag::value && SuperDimensionsTag::value <= MeshConfig::CellTopology::dimensions, "invalid superentity dimensions");

	typedef typename MeshTraits< MeshConfig >::template EntityTraits< SuperDimensionsTag::value >::EntityTopology SuperentityTopology;

	static const bool previousSuperDimensionsValue = MeshEntityOrientationNeeded< MeshConfig, DimensionsTag, typename SuperDimensionsTag::Decrement >::value;
	static const bool thisSuperDimensionsValue = MeshTraits< MeshConfig >::template SubentityTraits< SuperentityTopology, DimensionsTag::value >::orientationEnabled;

   public:
      static const bool value = ( previousSuperDimensionsValue || thisSuperDimensionsValue );
};

template< typename MeshConfig, typename DimensionsTag >
class MeshEntityOrientationNeeded< MeshConfig, DimensionsTag, DimensionsTag >
{
	static_assert( 0 <= DimensionsTag::value && DimensionsTag::value <= MeshConfig::CellTopology::dimensions, "invalid dimensions" );

   public:
      static const bool value = false;
};


template< typename MeshConfig,
          int Dimensions >
class MeshEntityTraits
{
   public:

      static const bool storageEnabled = MeshConfig::entityStorage( Dimensions );
      static const bool orientationNeeded = MeshEntityOrientationNeeded< MeshConfig, MeshDimensionsTag< Dimensions > >::value;

      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;
      typedef typename MeshEntityTopology< MeshConfig, Dimensions >::Topology   EntityTopology;
 
      typedef MeshEntity< MeshConfig, EntityTopology >                          EntityType;
      typedef MeshEntitySeed< MeshConfig, EntityTopology >                      SeedType;
      typedef MeshEntityReferenceOrientation< MeshConfig, EntityTopology >      ReferenceOrientationType;
      typedef MeshEntitySeedKey< MeshConfig, EntityTopology >                   Key;


      typedef Containers::Array< EntityType, Devices::Host, GlobalIndexType >               StorageArrayType;
      typedef Containers::SharedArray< EntityType, Devices::Host, GlobalIndexType >         AccessArrayType;
      typedef tnlIndexedSet< EntityType, GlobalIndexType, Key >                      UniqueContainerType;
      typedef tnlIndexedSet< SeedType, GlobalIndexType, Key >                        SeedIndexedSetType;
      typedef Containers::Array< SeedType, Devices::Host, GlobalIndexType >                 SeedArrayType;
      typedef Containers::Array< ReferenceOrientationType, Devices::Host, GlobalIndexType > ReferenceOrientationArrayType;

      typedef Containers::tnlConstSharedArray< EntityType, Devices::Host, GlobalIndexType >    SharedArrayType;
};

} // namespace Meshes
} // namespace TNL
