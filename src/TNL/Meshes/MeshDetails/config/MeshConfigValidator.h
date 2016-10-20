/***************************************************************************
                          MeshConfigValidator.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/MeshDimensionTag.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename dimensions >
class MeshConfigValidatorSubtopologyLayer
   : public MeshConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, typename dimensions::Decrement >
{
   static_assert( ! MeshConfig::subentityStorage( EntityTopology(), dimensions::value ) ||
                    MeshConfig::entityStorage( EntityTopology::dimensions ),
                  "entities of which subentities are stored must be stored" );
   static_assert( ! MeshConfig::subentityStorage( EntityTopology(), dimensions::value ) ||
                    MeshConfig::entityStorage( dimensions::value ),
                  "entities that are stored as subentities must be stored");
   static_assert( ! MeshConfig::subentityOrientationStorage( EntityTopology(), dimensions::value ) ||
                    MeshConfig::subentityStorage( EntityTopology(), dimensions::value ),
                  "orientation can be stored only for subentities that are stored");
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, MeshDimensionsTag< 0 > >
{
   static_assert( ! MeshConfig::subentityStorage( EntityTopology(), 0 ) ||
                    MeshConfig::entityStorage( 0 ),
                  "entities that are stored as subentities must be stored" );
   static_assert( ! MeshConfig::subentityOrientationStorage( EntityTopology(), 0 ),
                  "storage of vertex orientation does not make sense" );
};


template< typename MeshConfig,
          typename EntityTopology,
          typename dimensions >
class MeshConfigValidatorSupertopologyLayer
   : public MeshConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, typename dimensions::Decrement >
{
   static_assert( ! MeshConfig::superentityStorage( EntityTopology(), 0 ) ||
                    MeshConfig::entityStorage( EntityTopology::dimensions ),
                  "entities of which superentities are stored must be stored");
   static_assert( ! MeshConfig::superentityStorage( EntityTopology(), 0 ) ||
                    MeshConfig::entityStorage( dimensions::value ),
                  "entities that are stored as superentities must be stored");
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, MeshDimensionsTag< EntityTopology::dimensions > >
{};


template< typename MeshConfig, int dimensions >
class MeshConfigValidatorLayer
   : public MeshConfigValidatorLayer< MeshConfig, dimensions - 1 >,
     public MeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                 typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                 MeshDimensionsTag< dimensions - 1 > >,
     public MeshConfigValidatorSupertopologyLayer< MeshConfig,
                                                   typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology,
                                                   MeshDimensionsTag< MeshConfig::CellTopology::dimensions > >
{
   using Topology = typename MeshSubtopology< typename MeshConfig::CellTopology, dimensions >::Topology;

   static_assert( ! MeshConfig::entityStorage( dimensions ) ||
                    MeshConfig::subentityStorage( Topology(), 0 ),
                  "subvertices of all stored entities must be stored");
};

template< typename MeshConfig >
class MeshConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class MeshConfigValidatorLayerCell
   : public MeshConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimensions - 1 >,
     public MeshConfigValidatorSubtopologyLayer< MeshConfig,
                                                 typename MeshConfig::CellTopology,
                                                 MeshDimensionsTag< MeshConfig::CellTopology::dimensions - 1 > >
{
   using CellTopology = typename MeshConfig::CellTopology;
   static constexpr int dimensions = CellTopology::dimensions;

   static_assert( ! MeshConfig::entityStorage( dimensions ) ||
                    MeshConfig::subentityStorage( CellTopology(), 0 ),
                  "subvertices of all stored entities must be stored" );
};

template< typename MeshConfig >
class MeshConfigValidator
   : public MeshConfigValidatorLayerCell< MeshConfig >
{
   static constexpr int meshDimensions = MeshConfig::CellTopology::dimensions;

   static_assert( 1 <= meshDimensions, "zero dimensional meshes are not supported" );
   static_assert( meshDimensions <= MeshConfig::worldDimensions, "world dimension must not be less than mesh dimension" );

   static_assert( MeshConfig::entityStorage( 0 ), "mesh vertices must be stored" );
   static_assert( MeshConfig::entityStorage( meshDimensions ), "mesh cells must be stored" );
};

} // namespace Meshes
} // namespace TNL

