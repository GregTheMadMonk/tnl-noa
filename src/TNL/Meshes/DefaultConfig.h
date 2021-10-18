/***************************************************************************
                          DefaultConfig.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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

namespace TNL {
namespace Meshes {

/****
 * Basic structure for mesh configuration.
 */
template< typename Cell,
          int SpaceDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = short int >
struct DefaultConfig
{
   using CellTopology = Cell;
   using RealType = Real;
   using GlobalIndexType = GlobalIndex;
   using LocalIndexType = LocalIndex;

   static constexpr int spaceDimension = SpaceDimension;
   static constexpr int meshDimension = Cell::dimension;

   /****
    * Storage of subentities of mesh entities.
    */
   template< typename EntityTopology >
   static constexpr bool subentityStorage( EntityTopology, int SubentityDimension )
   {
      return true;
      // Subvertices must be stored for all entities which appear in other
      // subentity or superentity mappings.
      //return SubentityDimension == 0;
   }

   /****
    * Storage of superentities of mesh entities.
    */
   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology, int SuperentityDimension )
   {
      return true;
   }

   /****
    * Storage of mesh entity tags. Boundary tags are necessary for the mesh traverser.
    *
    * The configuration must satisfy the following necessary conditions in
    * order to provide boundary tags:
    *    - faces must store the cell indices in the superentity layer
    *    - if dim(entity) < dim(face), the entities on which the tags are stored
    *      must be stored as subentities of faces
    */
   template< typename EntityTopology >
   static constexpr bool entityTagsStorage( EntityTopology )
   {
      using FaceTopology = typename Topologies::Subtopology< CellTopology, meshDimension - 1 >::Topology;
      return superentityStorage( FaceTopology(), meshDimension ) &&
             ( EntityTopology::dimension >= meshDimension - 1 || subentityStorage( FaceTopology(), EntityTopology::dimension ) );
      //return false;
   }

   /****
    * Storage of the dual graph.
    *
    * If enabled, links from vertices to cells must be stored.
    */
   static constexpr bool dualGraphStorage()
   {
      return true;
   }

   /****
    * Cells must have at least this number of common vertices to be considered
    * as neighbors in the dual graph.
    */
   static constexpr int dualGraphMinCommonVertices = meshDimension;
};

} // namespace Meshes
} // namespace TNL
