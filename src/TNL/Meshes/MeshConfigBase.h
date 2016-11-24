/***************************************************************************
                          MeshConfigBase.h  -  description
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
#include <TNL/param-types.h>

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
struct MeshConfigBase
{
   typedef Cell        CellTopology;
   typedef Real        RealType;
   typedef GlobalIndex GlobalIndexType;
   typedef LocalIndex  LocalIndexType;
   typedef Id          IdType;

   static const int worldDimension = WorldDimension;
   static const int meshDimension = Cell::dimension;
 
   static_assert( worldDimension >= meshDimension, "The cell dimension cannot be larger than the world dimension." );
   static_assert( meshDimension > 0, "The cell dimension must be at least 1." );

   static String getType()
   {
      return String( "Meshes::MeshConfigBase< " ) +
             Cell::getType() + ", " +
             String( WorldDimension ) + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< GlobalIndex >() + ", " +
             TNL::getType< LocalIndex >() + ", " +
             TNL::getType< Id >() + " >";
   };
 
   /****
    * Storage of mesh entities.
    */
   static constexpr bool entityStorage( int dimension )
   {
      /****
       *  Vertices and cells must always be stored
       */
      return true;
      //return ( dimension == 0 || dimension == cellDimension );
   }
 
   /****
    *  Storage of subentities of mesh entities
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
    *  Storage of superentities of mesh entities
    */
   template< typename EntityTopology >
   static constexpr bool superentityStorage( EntityTopology, int SuperentityDimension )
   {
      return entityStorage( EntityTopology::dimension );
      //return false;
   }
};

} // namespace Meshes
} // namespace TNL
