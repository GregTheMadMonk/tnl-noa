/***************************************************************************
                          VerticesPerEntity.h  -  description
                             -------------------
    begin                : Mar 18, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Meshes/MeshEntity.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace details {

template< typename T, typename Enable = void >
struct has_entity_topology : std::false_type {};

template< typename T >
struct has_entity_topology< T, typename enable_if_type< typename T::EntityTopology >::type >
: std::true_type
{};

} // namespace details

template< typename Entity,
          bool _is_mesh_entity = details::has_entity_topology< Entity >::value >
struct VerticesPerEntity
{
   static constexpr int count = Topologies::Subtopology< typename Entity::EntityTopology, 0 >::count;
};

template< typename MeshConfig, typename Device >
struct VerticesPerEntity< MeshEntity< MeshConfig, Device, Topologies::Vertex >, true >
{
   static constexpr int count = 1;
};

template< typename GridEntity >
struct VerticesPerEntity< GridEntity, false >
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr int count =
      (dim == 0) ? 1 :
      (dim == 1) ? 2 :
      (dim == 2) ? 4 :
                   8;
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL
