/***************************************************************************
                          ConfigValidator.h  -  description
                             -------------------
    begin                : Aug 14, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {
namespace BoundaryTags {

template< typename MeshConfig,
          typename EntityTopology,
          bool BoundaryTagsStorage = MeshConfig::boundaryTagsStorage( EntityTopology() ) >
class ConfigValidatorBoundaryTagsLayer
{
   using FaceTopology = typename Topologies::Subtopology< typename MeshConfig::CellTopology, MeshConfig::meshDimension - 1 >::Topology;

   static_assert( MeshConfig::superentityStorage( FaceTopology(), MeshConfig::meshDimension ),
                  "Faces must store the cell superentity indices when any entity has boundary tags." );
   static_assert( EntityTopology::dimension >= MeshConfig::meshDimension - 1 || MeshConfig::subentityStorage( FaceTopology(), EntityTopology::dimension ),
                  "Faces must store the subentity indices of the entities on which the boundary tags are stored." );
};

template< typename MeshConfig,
          typename EntityTopology >
class ConfigValidatorBoundaryTagsLayer< MeshConfig, EntityTopology, false >
{
};


template< typename MeshConfig, int dimension = MeshConfig::meshDimension >
class ConfigValidatorLayer
   : public ConfigValidatorLayer< MeshConfig, dimension - 1 >,
     public ConfigValidatorBoundaryTagsLayer< MeshConfig,
                                              typename MeshTraits< MeshConfig >::template EntityTraits< dimension >::EntityTopology >
{
};

template< typename MeshConfig >
class ConfigValidatorLayer< MeshConfig, 0 >
{
};

template< typename MeshConfig >
class ConfigValidator
   : public ConfigValidatorLayer< MeshConfig >
{
};

} // namespace BoundaryTags
} // namespace Meshes
} // namespace TNL
