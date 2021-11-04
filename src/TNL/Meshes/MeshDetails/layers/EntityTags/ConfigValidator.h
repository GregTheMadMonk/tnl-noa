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
namespace EntityTags {

template< typename MeshConfig,
          int EntityDimension,
          bool entityTagsStorage = MeshConfig::entityTagsStorage( EntityDimension ) >
class ConfigValidatorEntityTagsLayer
{
   static_assert( MeshConfig::superentityStorage( MeshConfig::meshDimension - 1, MeshConfig::meshDimension ),
                  "Faces must store the cell superentity indices when any entity has boundary tags." );
   static_assert( EntityDimension >= MeshConfig::meshDimension - 1 || MeshConfig::subentityStorage( MeshConfig::meshDimension - 1, EntityDimension ),
                  "Faces must store the subentity indices of the entities on which the boundary tags are stored." );
};

template< typename MeshConfig,
          int EntityDimension >
class ConfigValidatorEntityTagsLayer< MeshConfig, EntityDimension, false >
{
};


template< typename MeshConfig, int dimension = MeshConfig::meshDimension >
class ConfigValidatorLayer
   : public ConfigValidatorLayer< MeshConfig, dimension - 1 >,
     public ConfigValidatorEntityTagsLayer< MeshConfig, dimension >
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

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
