/***************************************************************************
                          MeshConfigValidator.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshDimensionTag.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename MeshEntity,
          typename dimensions >
class MeshConfigValidatorSubtopologyLayer :
public MeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, typename dimensions::Decrement >
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
class MeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, MeshDimensionTag< 0 > >
{
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), 0 ) ||
                    MeshConfig::entityStorage( 0 ), "entities that are stored as subentities must be stored" );
   static_assert( ! MeshConfig::subentityOrientationStorage( MeshEntity(), 0 ), "storage of vertex orientation does not make sense" );
};


template< typename MeshConfig,
          typename MeshEntity,
          typename dimensions >
class MeshConfigValidatorSupertopologyLayer :
public MeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, typename dimensions::Decrement >
{
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) ||
                  MeshConfig::entityStorage( MeshEntity::dimensions ), "entities of which superentities are stored must be stored");
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) ||
                  MeshConfig::entityStorage( dimensions::value ), "entities that are stored as superentities must be stored");
};

template< typename MeshConfig,
          typename MeshEntity >
class MeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, MeshDimensionTag< MeshEntity::dimensions > >
{};


template< typename MeshConfig, int dimensions >
class MeshConfigValidatorLayer :
 public MeshConfigValidatorLayer< MeshConfig, dimensions - 1 >,
 public MeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                MeshDimensionTag< dimensions - 1 > >,
 public MeshConfigValidatorSupertopologyLayer< MeshConfig,
                                                  typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                  MeshDimensionTag< MeshConfig::CellTopology::dimensions > >
{
	typedef typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology Topology;

	static_assert( ! MeshConfig::entityStorage( dimensions ) || MeshConfig::subentityStorage( Topology(), 0 ), "subvertices of all stored entities must be stored");
};

template< typename MeshConfig >
class MeshConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class MeshConfigValidatorLayerCell :
   public MeshConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimensions - 1 >,
   public MeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                  typename MeshConfig::CellTopology,
                                                  MeshDimensionTag< MeshConfig::CellTopology::dimensions - 1 > >
{
	typedef typename MeshConfig::CellTopology    CellTopology;
 	static const int dimensions =  CellTopology::dimensions;

	static_assert( !MeshConfig::entityStorage( dimensions ) || MeshConfig::subentityStorage( CellTopology(), 0 ), "subvertices of all stored entities must be stored");
};

template<typename MeshConfig >
class MeshConfigValidator : public MeshConfigValidatorLayerCell< MeshConfig >
{
	static const int meshDimension = MeshConfig::CellTopology::dimensions;

	static_assert(1 <= meshDimension, "zero dimensional meshes are not supported");
	static_assert( meshDimension <= MeshConfig::worldDimension, "world dimension must not be less than mesh dimension");

	static_assert( MeshConfig::entityStorage( 0 ), "mesh vertices must be stored");
	static_assert( MeshConfig::entityStorage( meshDimension ), "mesh cells must be stored");
};

} // namespace Meshes
} // namespace TNL

