/***************************************************************************
                          tnlMeshSubentitySeedCreator.h  -  description
                             -------------------
    begin                : Aug 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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

#ifndef TNLMESHSUBENTITYSEEDCREATOR_H
#define	TNLMESHSUBENTITYSEEDCREATOR_H

#include <core/tnlStaticFor.h>

template< typename MeshConfig,
          typename EntityTopology,
          typename SubDimensionsTag >
class tnlMeshSubentitySeedsCreator
{
	typedef typename tnlMeshConfigTraits< MeshConfig >::LocalIndexType                                       LocalIndexType;
	typedef typename tnlMeshConfigTraits< MeshConfig >::template SubentityTraits< EntityTopology, SubDimensionsTag > SubentityTraits;
	typedef typename SubentityTraits::SubentityTag                                                              Subtopology;
	typedef typename tnlMeshConfigTraits< MeshConfig >::IdArrayAccessorType                                   IdArrayAccessorType;
	typedef typename tnlMeshConfigTraits< MeshConfig >::template SubentityTraits< Subtopology, tnlDimensionsTag< 0 > >     SubentityVertexTraits;

	static const LocalIndexType SUBENTITIES_COUNT = SubentityTraits::count;
	static const LocalIndexType SUBENTITY_VERTICES_COUNT = SubentityVertexTraits::count;

   public:
      typedef typename SubentityTraits::SeedArrayType SubentitySeedArray;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >  EntitySeed;
      //typedef typename tnlMeshEntitiesTraits< MeshConfig, SubDimensionsTag >::SeedIndexedSetType                     SeedIndexedSet;

      //template< typename SeedIndexedSet >
      static SubentitySeedArray create( const EntitySeed &entitySeed  )
      {
         SubentitySeedArray subentitySeeds;
         tnlStaticFor< LocalIndexType, 0, SUBENTITIES_COUNT, CreateSubentitySeeds >::exec( subentitySeeds, entitySeed.getCornerIds() );
         //tnlStaticFor< LocalIndexType, 0, SUBENTITIES_COUNT, CreateSubentitySeeds >::exec( indexedSet, entitySeed.getCornerIds() );
         
         return subentitySeeds;
      }

   private:
      typedef tnlMeshEntitySeed< MeshConfig, Subtopology > SubentitySeed;

      template< LocalIndexType subentityIndex >
      class CreateSubentitySeeds
      {
         public:
            static void exec( SubentitySeedArray &subentitySeeds, IdArrayAccessorType vertexIds )
            //static void exec( SeedIndexedSet& indexedSet, IdArrayAccessorType vertexIds )
            {
               //EntitySeed seed;
               tnlStaticFor< LocalIndexType, 0, SUBENTITY_VERTICES_COUNT, SetSubentitySeedVertex >::exec( subentitySeeds[ subentityIndex ], vertexIds );
               //indexedSet.insert( seed );
            }

         private:
            template< LocalIndexType subentityVertexIndex >
            class SetSubentitySeedVertex
            {
               public:
                  static void exec( SubentitySeed &subentitySeed, IdArrayAccessorType vertexIds )
                  {
                     static const LocalIndexType VERTEX_INDEX = SubentityTraits::template Vertex< subentityIndex, subentityVertexIndex >::index;
                     subentitySeed.setCornerId( subentityVertexIndex, vertexIds[ VERTEX_INDEX ] );
                  }
            };
      };
};

#endif	/* TNLMESHSUBENTITYSEEDCREATOR_H */

