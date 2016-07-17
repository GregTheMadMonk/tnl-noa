/***************************************************************************
                          tnlMeshConfigValidator.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlAssert.h>
#include <mesh/topologies/tnlMeshEntityTopology.h>
#include <mesh/tnlDimensionsTag.h>

namespace TNL {

template< typename MeshConfig,
          typename MeshEntity,
          typename dimensions >
class tnlMeshConfigValidatorSubtopologyLayer :
public tnlMeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, typename dimensions::Decrement >
{
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), dimensions::value ) ||
                    MeshConfig::entityStorage( MeshEntity::dimensions ), "entities of which subentities are stored must be stored" );
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), dimensions::value ) ||
                    MeshConfig::entityStorage( dimensions::value ), "entities that are stored as subentities must be stored");
   static_assert( ! MeshConfig::subentityOrientationStorage( MeshEntity(), dimensions::value ) ||
                    MeshConfig::subentityStorage( MeshEntity(), dimensions::value ), "orientation can be stored only for subentities that are stored");
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, tnlDimensionsTag< 0 > >
{
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), 0 ) ||
                    MeshConfig::entityStorage( 0 ), "entities that are stored as subentities must be stored" );
   static_assert( ! MeshConfig::subentityOrientationStorage( MeshEntity(), 0 ), "storage of vertex orientation does not make sense" );
};


template< typename MeshConfig,
          typename MeshEntity,
          typename dimensions >
class tnlMeshConfigValidatorSupertopologyLayer :
public tnlMeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, typename dimensions::Decrement >
{
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) ||
                  MeshConfig::entityStorage( MeshEntity::dimensions ), "entities of which superentities are stored must be stored");
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) ||
                  MeshConfig::entityStorage( dimensions::value ), "entities that are stored as superentities must be stored");
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, tnlDimensionsTag< MeshEntity::dimensions > >
{};


template< typename MeshConfig, int dimensions >
class tnlMeshConfigValidatorLayer :
 public tnlMeshConfigValidatorLayer< MeshConfig, dimensions - 1 >,
 public tnlMeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                typename tnlMeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                tnlDimensionsTag< dimensions - 1 > >,
 public tnlMeshConfigValidatorSupertopologyLayer< MeshConfig,
                                                  typename tnlMeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                  tnlDimensionsTag< MeshConfig::CellTopology::dimensions > >
{
	typedef typename tnlMeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology Topology;

	static_assert( ! MeshConfig::entityStorage( dimensions ) || MeshConfig::subentityStorage( Topology(), 0 ), "subvertices of all stored entities must be stored");
};

template< typename MeshConfig >
class tnlMeshConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class tnlMeshConfigValidatorLayerCell :
   public tnlMeshConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimensions - 1 >,
   public tnlMeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                  typename MeshConfig::CellTopology,
                                                  tnlDimensionsTag< MeshConfig::CellTopology::dimensions - 1 > >
{
	typedef typename MeshConfig::CellTopology    CellTopology;
 	static const int dimensions =  CellTopology::dimensions;

	static_assert( !MeshConfig::entityStorage( dimensions ) || MeshConfig::subentityStorage( CellTopology(), 0 ), "subvertices of all stored entities must be stored");
};

template<typename MeshConfig >
class tnlMeshConfigValidator : public tnlMeshConfigValidatorLayerCell< MeshConfig >
{
	static const int meshDimensions = MeshConfig::CellTopology::dimensions;

	static_assert(1 <= meshDimensions, "zero dimensional meshes are not supported");
	static_assert( meshDimensions <= MeshConfig::worldDimensions, "world dimension must not be less than mesh dimension");

	static_assert( MeshConfig::entityStorage( 0 ), "mesh vertices must be stored");
	static_assert( MeshConfig::entityStorage( meshDimensions ), "mesh cells must be stored");
};

} // namespace TNL

