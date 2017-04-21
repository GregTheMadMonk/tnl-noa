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
//#include <TNL/Containers/ConstSharedArray.h>
#include <TNL/Containers/IndexedSet.h>
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
          typename DimensionTag,
          typename SuperDimensionTag = MeshDimensionTag< MeshConfig::meshDimension > >
class MeshEntityOrientationNeeded
{
	static_assert( 0 <= DimensionTag::value && DimensionTag::value < MeshConfig::CellTopology::dimensions, "invalid dimensions" );
	static_assert( DimensionTag::value < SuperDimensionTag::value && SuperDimensionTag::value <= MeshConfig::CellTopology::dimensions, "invalid superentity dimension");

	typedef typename MeshTraits< MeshConfig >::template EntityTraits< SuperDimensionTag::value >::EntityTopology SuperentityTopology;

	static const bool previousSuperDimensionValue = MeshEntityOrientationNeeded< MeshConfig, DimensionTag, typename SuperDimensionTag::Decrement >::value;
	static const bool thisSuperDimensionValue = MeshTraits< MeshConfig >::template SubentityTraits< SuperentityTopology, DimensionTag::value >::orientationEnabled;

   public:
      static const bool value = ( previousSuperDimensionValue || thisSuperDimensionValue );
};

template< typename MeshConfig, typename DimensionTag >
class MeshEntityOrientationNeeded< MeshConfig, DimensionTag, DimensionTag >
{
	static_assert( 0 <= DimensionTag::value && DimensionTag::value <= MeshConfig::CellTopology::dimensions, "invalid dimensions" );

   public:
      static const bool value = false;
};


template< typename MeshConfig,
          int Dimension >
class MeshEntityTraits
{
   public:

      static const bool storageEnabled = MeshConfig::entityStorage( Dimension );
      static const bool orientationNeeded = MeshEntityOrientationNeeded< MeshConfig, MeshDimensionTag< Dimension > >::value;

      typedef typename MeshConfig::GlobalIndexType                                 GlobalIndexType;
      typedef typename MeshConfig::LocalIndexType                                  LocalIndexType;
      typedef typename MeshEntityTopology< MeshConfig, Dimension >::Topology   EntityTopology;
 
      typedef MeshEntity< MeshConfig, EntityTopology >                          EntityType;
      typedef MeshEntitySeed< MeshConfig, EntityTopology >                      SeedType;
      typedef MeshEntityReferenceOrientation< MeshConfig, EntityTopology >      ReferenceOrientationType;
      typedef MeshEntitySeedKey< MeshConfig, EntityTopology >                   Key;


      typedef Containers::Array< EntityType, Devices::Host, GlobalIndexType >               StorageArrayType;
      typedef Containers::SharedArray< EntityType, Devices::Host, GlobalIndexType >         AccessArrayType;
      typedef Containers::IndexedSet< EntityType, GlobalIndexType, Key >                      UniqueContainerType;
      typedef Containers::IndexedSet< SeedType, GlobalIndexType, Key >                        SeedIndexedSetType;
      typedef Containers::Array< SeedType, Devices::Host, GlobalIndexType >                 SeedArrayType;
      typedef Containers::Array< ReferenceOrientationType, Devices::Host, GlobalIndexType > ReferenceOrientationArrayType;

      typedef Containers::tnlConstSharedArray< EntityType, Devices::Host, GlobalIndexType >    SharedArrayType;
};

} // namespace Meshes
} // namespace TNL
