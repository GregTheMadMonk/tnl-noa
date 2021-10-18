/***************************************************************************
                          ConfigValidator.h  -  description
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

#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/DimensionTag.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class ConfigValidatorSubtopologyLayer
   : public ConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   static_assert( ! MeshConfig::subentityStorage( EntityTopology(), DimensionTag::value ) ||
                    MeshConfig::subentityStorage( EntityTopology(), 0 ),
                  "entities of which subentities are stored must store their subvertices" );
   static_assert( ! MeshConfig::subentityStorage( EntityTopology(), DimensionTag::value ) ||
                    MeshConfig::subentityStorage( EntityTopology(), 0 ),
                  "entities that are stored as subentities must store their subvertices" );
};

template< typename MeshConfig,
          typename EntityTopology >
class ConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, DimensionTag< 0 > >
{};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class ConfigValidatorSupertopologyLayer
   : public ConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   static_assert( ! MeshConfig::superentityStorage( EntityTopology(), DimensionTag::value ) ||
                    MeshConfig::subentityStorage( EntityTopology(), 0 ),
                  "entities of which superentities are stored must store their subvertices");
   static_assert( ! MeshConfig::superentityStorage( EntityTopology(), DimensionTag::value ) ||
                    MeshConfig::subentityStorage( EntityTopology(), 0 ),
                  "entities that are stored as superentities must store their subvertices");
};

template< typename MeshConfig,
          typename EntityTopology >
class ConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, DimensionTag< EntityTopology::dimension > >
{};


template< typename MeshConfig, int dimension >
class ConfigValidatorLayer
   : public ConfigValidatorLayer< MeshConfig, dimension - 1 >,
     public ConfigValidatorSubtopologyLayer< MeshConfig,
                                             typename Topologies::Subtopology< typename MeshConfig::CellTopology, dimension >::Topology,
                                             DimensionTag< dimension - 1 > >,
     public ConfigValidatorSupertopologyLayer< MeshConfig,
                                               typename Topologies::Subtopology< typename MeshConfig::CellTopology, dimension >::Topology,
                                               DimensionTag< MeshConfig::CellTopology::dimension > >
{};

template< typename MeshConfig >
class ConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class ConfigValidatorLayerCell
   : public ConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimension - 1 >,
     public ConfigValidatorSubtopologyLayer< MeshConfig,
                                             typename MeshConfig::CellTopology,
                                             DimensionTag< MeshConfig::CellTopology::dimension - 1 > >
{
   using CellTopology = typename MeshConfig::CellTopology;

   static_assert( MeshConfig::subentityStorage( CellTopology(), 0 ),
                  "subvertices of cells must be stored" );
};

template< typename MeshConfig >
class ConfigValidator
   : public ConfigValidatorLayerCell< MeshConfig >
{
   static constexpr int meshDimension = MeshConfig::CellTopology::dimension;

   static_assert( 1 <= meshDimension, "zero dimensional meshes are not supported" );
   static_assert( meshDimension <= MeshConfig::spaceDimension, "space dimension must not be less than mesh dimension" );
};

} // namespace Meshes
} // namespace TNL
