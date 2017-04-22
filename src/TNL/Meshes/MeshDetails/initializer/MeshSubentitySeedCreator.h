/***************************************************************************
                          MeshSubentitySeedCreator.h  -  description
                             -------------------
    begin                : Aug 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/StaticFor.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SubDimensionTag >
class MeshSubentitySeedsCreator
{
	typedef typename MeshTraits< MeshConfig >::LocalIndexType                                                      LocalIndexType;
	typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, SubDimensionTag::value > SubentityTraits;
	typedef typename SubentityTraits::SubentityTopology                                                               SubentityTopology;
	typedef typename MeshTraits< MeshConfig >::IdArrayAccessorType                                                 IdArrayAccessorType;
	typedef typename MeshTraits< MeshConfig >::template SubentityTraits< SubentityTopology, 0 >                    SubentityVertexTraits;

	static const LocalIndexType SUBENTITIES_COUNT = SubentityTraits::count;
	static const LocalIndexType SUBENTITY_VERTICES_COUNT = SubentityVertexTraits::count;

   public:
      typedef typename SubentityTraits::SeedArrayType SubentitySeedArray;
      typedef MeshEntitySeed< MeshConfig, EntityTopology >  EntitySeed;
      //typedef typename MeshEntityTraits< MeshConfig, SubDimensionTag >::SeedIndexedSetType                     SeedIndexedSet;

      //template< typename SeedIndexedSet >
      static SubentitySeedArray create( const EntitySeed &entitySeed  )
      {
         SubentitySeedArray subentitySeeds;
         StaticFor< LocalIndexType, 0, SUBENTITIES_COUNT, CreateSubentitySeeds >::exec( subentitySeeds, entitySeed.getCornerIds() );
         //StaticFor< LocalIndexType, 0, SUBENTITIES_COUNT, CreateSubentitySeeds >::exec( indexedSet, entitySeed.getCornerIds() );
 
         return subentitySeeds;
      }

   private:
      typedef MeshEntitySeed< MeshConfig, SubentityTopology > SubentitySeed;

      template< LocalIndexType subentityIndex >
      class CreateSubentitySeeds
      {
         public:
            static void exec( SubentitySeedArray &subentitySeeds, IdArrayAccessorType vertexIds )
            //static void exec( SeedIndexedSet& indexedSet, IdArrayAccessorType vertexIds )
            {
               //EntitySeed seed;
               StaticFor< LocalIndexType, 0, SUBENTITY_VERTICES_COUNT, SetSubentitySeedVertex >::exec( subentitySeeds[ subentityIndex ], vertexIds );
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

} // namespace Meshes
} // namespace TNL

