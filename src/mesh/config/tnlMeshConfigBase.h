/***************************************************************************
                          tnlMeshConfigBase.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

/****
 * Basic structure for mesh configuration.
 * Setting Id to GlobalIndex enables storage of entity Id.
 * It means that each mesh entity stores its index in its
 * mesh storage layer.
 */
template< typename Cell,
          int WorldDimensions = Cell::dimensions,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex,
          typename Id = void >
struct tnlMeshConfigBase
{
   typedef Cell        CellTopology;
   typedef Real        RealType;
   typedef GlobalIndex GlobalIndexType;
   typedef LocalIndex  LocalIndexType;
   typedef Id          IdType;

   static const int worldDimensions = WorldDimensions;
   static const int meshDimensions = Cell::dimensions;

   static tnlString getType()
   {
      return tnlString( "tnlMeshConfigBase< >");
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
		//return ( dimensions == 0 || dimensions == cellDimensions );
	}
 
   /****
    *  Storage of subentities of mesh entities
    */
	template< typename MeshEntity >
	static constexpr bool subentityStorage( MeshEntity, int SubentityDimensions )
	{
      /****
       *  Vertices must always be stored
       */
      return true;
		//return ( SubentityDimensions == 0 );
	}

	/****
    * Storage of subentity orientations of mesh entities.
    * It must be false for vertices and cells.
    */
	template< typename MeshEntity >
	static constexpr bool subentityOrientationStorage( MeshEntity, int SubentityDimensions )
	{
		return ( SubentityDimensions > 0 );
	}

	/****
    *  Storage of superentities of mesh entities
    */
	template< typename MeshEntity >
	static constexpr bool superentityStorage( MeshEntity, int SuperentityDimensions )
	{
      return true;
		//return false;
	}
 
   static_assert( WorldDimensions >= Cell::dimensions, "The number of the cell dimensions cannot be larger than the world dimension." );
};

} // namespace TNL
