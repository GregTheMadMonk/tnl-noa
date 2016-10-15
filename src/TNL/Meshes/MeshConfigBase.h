/***************************************************************************
                          MeshConfigBase.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

namespace TNL {
namespace Meshes {

/****
 * Basic structure for mesh configuration.
 * Setting Id to GlobalIndex enables storage of entity Id.
 * It means that each mesh entity stores its index in its
 * mesh storage layer.
 */
template< typename Cell,
          int WorldDimension = Cell::dimensions,
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

   static const int worldDimensions = WorldDimensions;
   static const int meshDimensions = Cell::dimensions;
 
   static_assert( worldDimensions >= meshDimensions, "The cell dimension cannot be larger than the world dimension." );

   static String getType()
   {
      return String( "MeshConfigBase< >");
   };
 
   /****
    * Storage of mesh entities.
    */
	static constexpr bool entityStorage( int dimensions )
	{
      /****
       *  Vertices and cells must always be stored
       */
      return true;
		//return ( dimensions == 0 || dimensions == cellDimension );
	}
 
   /****
    *  Storage of subentities of mesh entities
    */
	template< typename MeshEntity >
	static constexpr bool subentityStorage( MeshEntity, int SubentityDimension )
	{
      /****
       *  Vertices must always be stored
       */
      return true;
		//return ( SubentityDimension == 0 );
	}

	/****
    * Storage of subentity orientations of mesh entities.
    * It must be false for vertices and cells.
    */
	template< typename MeshEntity >
	static constexpr bool subentityOrientationStorage( MeshEntity, int SubentityDimension )
	{
		return ( SubentityDimension > 0 );
	}

	/****
    *  Storage of superentities of mesh entities
    */
	template< typename MeshEntity >
	static constexpr bool superentityStorage( MeshEntity, int SuperentityDimension )
	{
      return true;
		//return false;
	}
};

} // namespace Meshes
} // namespace TNL
