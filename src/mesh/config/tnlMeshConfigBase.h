/***************************************************************************
                          tnlMeshConfigBase.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHCONFIGBASE_H_
#define TNLMESHCONFIGBASE_H_

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
   static const int cellDimensions = Cell::dimensions;

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

#ifdef UNDEF
/****
 * Explicit storage of all mesh entities by default.
 * To disable it, write your own specialization with given
 * dimensions and config tag.
 */
template< typename ConfigTag,
          int Dimensions >
struct tnlMeshEntityStorage
{
   enum { enabled = true };
};

/****
 * By default, ALL SUBENTITIES of a mesh entity ARE STORED
 * provided that they are stored in the mesh.
 * Write your own specialization if you do not want so.
 */
template< typename ConfigTag,
          typename EntityTag,
          int Dimensions >
struct tnlMeshSubentityStorage
{
   enum { enabled = tnlMeshEntityStorage< ConfigTag, Dimensions >::enabled };
};

/***
 * By default, NO SUPERENTITIES of any mesh entity ARE STORED.
 * Write your own specialization if you need to stored them.
 */
template< typename ConfigTag,
          typename EntityTag,
          int Dimensions >
struct tnlMeshSuperentityStorage
{
   enum { enabled = false };
};

#endif

#endif /* TNLMESHCONFIGBASE_H_ */
