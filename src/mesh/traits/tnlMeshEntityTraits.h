/***************************************************************************
                          tnlMeshEntityTraits.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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


      typedef tnlArray< EntityType, tnlHost, GlobalIndexType >                     StorageArrayType;
      typedef tnlSharedArray< EntityType, tnlHost, GlobalIndexType >               AccessArrayType;
      typedef tnlIndexedSet< EntityType, GlobalIndexType, Key >                    UniqueContainerType;
      typedef tnlIndexedSet< SeedType, GlobalIndexType, Key >                      SeedIndexedSetType;
      typedef tnlArray< SeedType, tnlHost, GlobalIndexType >                       SeedArrayType;
      typedef tnlArray< ReferenceOrientationType, tnlHost, GlobalIndexType >       ReferenceOrientationArrayType;

      typedef tnlConstSharedArray< EntityType, tnlHost, GlobalIndexType >          SharedArrayType;
};


#endif /* TNLMESHENTITYTRAITS_H_ */
