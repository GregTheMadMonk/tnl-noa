/***************************************************************************
                          tnlMeshConfigValidator.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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

#ifndef TNLMESHCONFIGVALIDATOR_H
#define	TNLMESHCONFIGVALIDATOR_H

#include <core/tnlAssert.h>

/*
template< typename MeshConfig,
          typename MeshEntity,
          int dimensions >
class tnlMeshConfigValidatorSubtopologyLayer :
   public tnlMeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, dimensions - 1 >
{
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), dimensions ) || 
                    MeshConfig::entityStorage( MeshEntity::dimensions ), "entities of which subentities are stored must be stored" );
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), dimensions ) ||
                    MeshConfig::entityStorage( dimensions ), "entities that are stored as subentities must be stored");
   // TODO: fix this 
   //static_assert( ! MeshConfig::subentityOrientationStorage(TTopology(), TDim()) || TConfig::subentityStorage(TTopology(), TDim()), "orientation can be stored only for subentities that are stored");
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshConfigValidatorSubtopologyLayer< MeshConfig, MeshEntity, 0 >
{
   static_assert( ! MeshConfig::subentityStorage( MeshEntity(), 0 ) ||
                    MeshConfig::entityStorage( 0 ), "entities that are stored as subentities must be stored" );
   static_assert( ! MeshConfig::subentityOrientationStorage( MeshEntity(), 0 ), "storage of vertex orientation does not make sense" );
};


template< typename MeshConfig,
          typename MeshEntity,
          int dimensions >
class tnlMeshConfigValidatorSupertopologyLayer :
 public tnlMeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, dimensions - 1 >
{
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) || MeshConfig::entityStorage( MeshEntity::dimensions ), "entities of which superentities are stored must be stored");
   static_assert( ! MeshConfig::superentityStorage( MeshEntity(), 0 ) || MeshConfig::entityStorage( dimensions ), "entities that are stored as superentities must be stored");
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshConfigValidatorSupertopologyLayer< MeshConfig, MeshEntity, MeshEntity::dimensions >
{};
*/

template< typename MeshConfig, int dimensions >
class tnlMeshConfigValidatorLayer :
 public tnlMeshConfigValidatorLayer< MeshConfig, dimensions - 1 >//,
 //public ConfigValidatorSubtopologyLayer< MeshConfig, typename Subtopology<typename TConfig::TCellTopology, TDim::VALUE>::TTopology, typename TDim::Decrement>,
 //public ConfigValidatorSupertopologyLayer< MeshConfig, typename Subtopology<typename TConfig::TCellTopology, TDim::VALUE>::TTopology, Dim<TConfig::TCellTopology::DIMENSION>>
{
	//typedef typename Subtopology<typename TConfig::TCellTopology, TDim::VALUE>::TTopology TTopology;

//	static_assert( ! MeshConfig::entityStorage( dimensions ) || MeshConfig::subentityStorage( TTopology(), Dim<0>()), "subvertices of all stored entities must be stored");
};

template< typename MeshConfig >
class tnlMeshConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class tnlMeshConfigValidatorLayerCell :
   public tnlMeshConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimensions - 1 >//,
//   public tnlMeshConfigValidatorSubtopologyLayer< MeshConfig, typename MeshConfig::CellType, MeshConfig::CellType::dimensions - 1 >
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


#endif	/* TNLMESHCONFIGVALIDATOR_H */

