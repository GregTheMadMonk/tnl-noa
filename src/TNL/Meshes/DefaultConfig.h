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

#include <TNL/String.h>
#include <TNL/TypeInfo.h>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>

namespace TNL {
namespace Meshes {

/****
 * Basic structure for mesh configuration.
 * Setting Id to GlobalIndex enables storage of entity Id.
 * It means that each mesh entity stores its index in its
 * mesh storage layer.
 */
template< typename Cell,
          int WorldDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex,
          typename Id = void >
struct DefaultConfig
{
   using CellTopology = Cell;
   using RealType = Real;
   using GlobalIndexType = GlobalIndex;
   using LocalIndexType = LocalIndex;
   using IdType = Id;

   static constexpr int worldDimension = WorldDimension;
   static constexpr int meshDimension = Cell::dimension;

   /****
    * Storage of mesh entities.
    */
   static constexpr bool entityStorage( int dimension )
   {
      /****
       * Vertices and cells must always be stored.
       */
      return true;
      //return ( dimension == 0 || dimension == cellDimension );
   }

   /****
    * Storage of subentities of mesh entities.
    */
   template< typename EntityTopology >
   static constexpr bool subentityStorage( EntityTopology, int SubentityDimension )
   {
      /****
       *  Subvertices of all stored entities must always be stored
       */
      return entityStorage( EntityTopology::dimension );
      //return entityStorage( EntityTopology::dimension ) &&
      //       SubentityDimension == 0;
   }

   /****
    * Storage of subentity orientations of mesh entities.
    * It must be false for vertices and cells.
    */
   template< typename EntityTopology >
   static constexpr bool subentityOrientationStorage( EntityTopology, int SubentityDimension )
   {
      return ( SubentityDimension > 0 );
   }

   /****
    * Storage of superentities of mesh entities.
    */
   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology, int SuperentityDimension )
   {
      return entityStorage( EntityTopology::dimension );
      //return false;
   }

   /****
    * Storage of boundary tags of mesh entities. Necessary for the mesh traverser.
    *
    * The configuration must satisfy the following necessary conditions in
    * order to provide boundary tags:
    *    - faces must be stored
    *    - faces must store the cell indices in the superentity layer
    *    - if dim(entity) < dim(face), the entities on which the tags are stored
    *      must be stored as subentities of faces
    * TODO: check this in the ConfigValidator
    */
   template< typename EntityTopology >
   static constexpr bool boundaryTagsStorage( EntityTopology )
   {
      using FaceTopology = typename Topologies::Subtopology< CellTopology, meshDimension - 1 >::Topology;
      return entityStorage( meshDimension - 1 ) &&
             superentityStorage( FaceTopology(), meshDimension ) &&
             ( EntityTopology::dimension >= meshDimension - 1 || subentityStorage( FaceTopology(), EntityTopology::dimension ) );
      //return false;
   }
};

} // namespace Meshes
} // namespace TNL
