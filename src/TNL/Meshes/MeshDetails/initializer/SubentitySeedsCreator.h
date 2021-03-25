/***************************************************************************
                          SubentitySeedsCreator.h  -  description
                             -------------------
    begin                : Aug 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/Polygon.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SubentityDimensionTag >
class SubentitySeedsCreator
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubvertexAccessorType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension, SubentityDimensionTag::value >::RowView;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubentityDimensionTag::value >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

   static constexpr LocalIndexType SUBENTITY_VERTICES_COUNT = MeshTraitsType::template SubentityTraits< SubentityTopology, 0 >::count;

public:
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, EntitySeed< MeshConfig, SubentityTopology > >;

   static SubentitySeedArray create( const SubvertexAccessorType& subvertices )
   {
      SubentitySeedArray subentitySeeds;
      Algorithms::staticFor< LocalIndexType, 0, SubentitySeedArray::getSize() >(
         [&] ( auto subentityIndex ) {
            Algorithms::staticFor< LocalIndexType, 0, SUBENTITY_VERTICES_COUNT >(
               [&] ( auto subentityVertexIndex ) {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  static constexpr LocalIndexType VERTEX_INDEX = SubentityTraits::template Vertex< decltype(subentityIndex){}, subentityVertexIndex >::index;
                  subentitySeeds[ subentityIndex ].setCornerId( subentityVertexIndex, subvertices.getColumnIndex( VERTEX_INDEX ) );
               }
            );
         }
      );

      return subentitySeeds;
   }

   static LocalIndexType getSubentitiesCount( const SubvertexAccessorType& subvertices )
   {
      return SubentityTraits::count;
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 1 > >
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubvertexAccessorType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension, 1 >::RowView;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::Array< SubentitySeed, DeviceType, LocalIndexType >;
   
   static SubentitySeedArray create( const SubvertexAccessorType& subvertices )
   {
      SubentitySeedArray seeds;
      LocalIndexType verticesCount = getVerticesCount( subvertices );
      seeds.setSize( verticesCount );

      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
      {
         SubentitySeed& seed = seeds[ i ];
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         seed.setCornerId( 1, subvertices.getColumnIndex( (i + 1) % verticesCount ) );
      }

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( const SubvertexAccessorType& subvertices )
   {
      return getVerticesCount( subvertices );
   }
private:
   static LocalIndexType getVerticesCount( const SubvertexAccessorType& subvertices )
   {
      LocalIndexType i;
      for( i = 0; i < subvertices.getSize() && subvertices.getColumnIndex( i ) >= 0; i++ ) {}
      return i;
   }
};

template< typename MeshConfig,
          typename EntityTopology >
class SubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag< 0 > >
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubvertexAccessorType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension, 0 >::RowView;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, EntitySeed< MeshConfig, SubentityTopology > >;

   static SubentitySeedArray create( const SubvertexAccessorType& subvertices )
   {
      SubentitySeedArray seeds;
      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );
      return seeds;
   }

   static LocalIndexType getSubentitiesCount( const SubvertexAccessorType& subvertices )
   {
      return subvertices.getSize();
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 0 > >
{
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubvertexAccessorType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension, 0 >::RowView;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeedArray = Containers::Array< EntitySeed< MeshConfig, SubentityTopology >, DeviceType, LocalIndexType >;

   static SubentitySeedArray create( const SubvertexAccessorType& subvertices )
   {
      SubentitySeedArray seeds;
      seeds.setSize( getVerticesCount( subvertices ) );

      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( const SubvertexAccessorType& subvertices )
   {
      return getVerticesCount( subvertices );
   }
private:
   static LocalIndexType getVerticesCount( const SubvertexAccessorType& subvertices )
   {
      LocalIndexType i;
      for( i = 0; i < subvertices.getSize() && subvertices.getColumnIndex( i ) >= 0; i++ ) {}
      return i;
   }
};

} // namespace Meshes
} // namespace TNL
