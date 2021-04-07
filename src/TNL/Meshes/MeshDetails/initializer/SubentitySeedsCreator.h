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
#include <TNL/Meshes/Topologies/SubentityVertexCount.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SubentityDimensionTag >
class SubentitySeedsCreator
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubentityDimensionTag::value >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, EntitySeed< MeshConfig, SubentityTopology > >;

   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray subentitySeeds;
      Algorithms::staticFor< LocalIndexType, 0, SubentitySeedArray::getSize() >(
         [&] ( auto subentityIndex ) {
            constexpr LocalIndexType subentityVerticesCount = Topologies::SubentityVertexCount< EntityTopology, SubentityTopology, subentityIndex >::count;
            auto& subentitySeed = subentitySeeds[ subentityIndex ];
            subentitySeed.setCornersCount( subentityVerticesCount );
            Algorithms::staticFor< LocalIndexType, 0, subentityVerticesCount >(
               [&] ( auto subentityVertexIndex ) {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  static constexpr LocalIndexType VERTEX_INDEX = SubentityTraits::template Vertex< decltype(subentityIndex){}, subentityVertexIndex >::index;
                  subentitySeed.setCornerId( subentityVertexIndex, subvertices.getColumnIndex( VERTEX_INDEX ) );
               }
            );
         }
      );

      return subentitySeeds;
   }

   constexpr static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 1 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::Array< SubentitySeed, DeviceType, LocalIndexType >;
   
   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      SubentitySeedArray seeds;
      seeds.setSize( subverticesCount );

      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
      {
         SubentitySeed& seed = seeds[ i ];
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         seed.setCornerId( 1, subvertices.getColumnIndex( (i + 1) % subverticesCount ) );
      }

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 2 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polyhedron;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 2 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::Array< SubentitySeed, DeviceType, LocalIndexType >;
   
   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& cellSeeds = initializer.getCellSeeds();
      const auto& faces = cellSeeds[ entityIndex ].getCornerIds();

      SubentitySeedArray seeds;
      seeds.setSize( faces.getSize() );

      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
      {
         SubentitySeed& seed = seeds[ i ];
         GlobalIndexType faceIdx = faces[ i ];
         const auto& subvertices = mesh.template getSubentitiesMatrix< 2, 0 >().getRow( faceIdx );
         const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< 2, 0 >( faceIdx );
         seed.setCornersCount( subverticesCount );
         for( LocalIndexType j = 0; j < subverticesCount; j++ )
         {
            seed.setCornerId( j, subvertices.getColumnIndex( j ) );
         }
      }

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      auto& cellSeeds = initializer.getCellSeeds();
      return cellSeeds[ entityIndex ].getCornersCount();
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 1 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polyhedron;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;
   using SeedSet               = typename MeshTraitsType::template EntityTraits< 1 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 1 > >;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::Array< SubentitySeed, DeviceType, LocalIndexType >;
   
   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SubentitySeedArray seeds;
      LocalIndexType subentitiesCount = getSubentitiesCount( initializer, mesh, entityIndex );
      seeds.setSize( subentitiesCount );
      GlobalIndexType seedsSize = 0;

      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ )
      {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         auto faceSubentitySeeds = FaceSubentitySeedsCreator::create( initializer, mesh, faceIdx );
         for( LocalIndexType j = 0; j < faceSubentitySeeds.getSize(); j++ )
         {
            const auto& faceSubentitySeed = faceSubentitySeeds[ j ];
            bool inserted = seedSet.insert( faceSubentitySeed ).second;
            if( inserted )
               seeds[ seedsSize++ ] = faceSubentitySeed;
         }
      }

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );
      for( LocalIndexType i = 0; i < facesCount; i++ )
      {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         auto faceSubentitySeeds = FaceSubentitySeedsCreator::create( initializer, mesh, faceIdx );
         for( LocalIndexType j = 0; j < faceSubentitySeeds.getSize(); j++ )
               seedSet.insert( faceSubentitySeeds[ j ] );
      }

      return seedSet.size();
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polyhedron;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;
   using SeedSet               = typename MeshTraitsType::template EntityTraits< 0 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 0 > >;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::Array< SubentitySeed, DeviceType, LocalIndexType >;
   
   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SubentitySeedArray seeds;
      LocalIndexType subentitiesCount = getSubentitiesCount( initializer, mesh, entityIndex );
      seeds.setSize( subentitiesCount );
      GlobalIndexType seedsSize = 0;

      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ )
      {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         auto faceSubentitySeeds = FaceSubentitySeedsCreator::create( initializer, mesh, faceIdx );
         for( LocalIndexType j = 0; j < faceSubentitySeeds.getSize(); j++ )
         {
            auto& faceSubentitySeed = faceSubentitySeeds[ j ];
            bool inserted = seedSet.insert( faceSubentitySeed ).second;
            if( inserted )
               seeds[ seedsSize++ ] = faceSubentitySeed;
         }
      }

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ )
      {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         auto faceSubentitySeeds = FaceSubentitySeedsCreator::create( initializer, mesh, faceIdx );
         for( LocalIndexType j = 0; j < faceSubentitySeeds.getSize(); j++ )
               seedSet.insert( faceSubentitySeeds[ j ] );
      }

      return seedSet.size();
   }
};

template< typename MeshConfig,
          typename EntityTopology >
class SubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, EntitySeed< MeshConfig, SubentityTopology > >;

   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray seeds;
      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );
      return seeds;
   }

   constexpr static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeedArray = Containers::Array< EntitySeed< MeshConfig, SubentityTopology >, Devices::Host, LocalIndexType >;

   static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      SubentitySeedArray seeds;
      seeds.setSize( subverticesCount );

      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );

      return seeds;
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

} // namespace Meshes
} // namespace TNL
