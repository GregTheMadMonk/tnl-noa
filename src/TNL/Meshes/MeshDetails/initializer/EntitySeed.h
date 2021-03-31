/***************************************************************************
                          EntitySeed.h  -  description
                             -------------------
    begin                : Aug 18, 2015
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

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/Polygon.h>

namespace TNL {
namespace Meshes {

template< typename EntitySeed >
struct EntitySeedHash;
template< typename EntitySeed >
struct EntitySeedEq;

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeed
{
   using MeshConfigTraits = MeshTraits< MeshConfig >;
   using SubvertexTraits = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 >;

   public:
      using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
      using LocalIndexType  = typename MeshTraits< MeshConfig >::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< SubvertexTraits::count, GlobalIndexType >;
      using HashType        = EntitySeedHash< EntitySeed >;
      using KeyEqual        = EntitySeedEq< EntitySeed >;

      //this function is here only for compatibility with MeshReader
      void setCornersCount( const LocalIndexType& cornersCount ) {}

      static constexpr LocalIndexType getCornersCount()
      {
         return SubvertexTraits::count;
      }

      void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
         TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

         this->cornerIds[ cornerIndex ] = pointIndex;
      }

      IdArrayType& getCornerIds()
      {
         return cornerIds;
      }

      const IdArrayType& getCornerIds() const
      {
         return cornerIds;
      }

   private:
      IdArrayType cornerIds;
};

template< typename MeshConfig >
class EntitySeed< MeshConfig, Topologies::Vertex >
{
   using MeshConfigTraits = MeshTraits< MeshConfig >;

   public:
      using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
      using LocalIndexType  = typename MeshTraits< MeshConfig >::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< 1, GlobalIndexType >;
      using HashType        = EntitySeedHash< EntitySeed >;
      using KeyEqual        = EntitySeedEq< EntitySeed >;

      //this function is here only for compatibility with MeshReader
      void setCornersCount( const LocalIndexType& cornersCount ) {}

      static constexpr LocalIndexType getCornersCount()
      {
         return 1;
      }

      void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_EQ( cornerIndex, 0, "corner index must be 0" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

         this->cornerIds[ cornerIndex ] = pointIndex;
      }

      IdArrayType& getCornerIds()
      {
         return cornerIds;
      }

      const IdArrayType& getCornerIds() const
      {
         return cornerIds;
      }

   private:
      IdArrayType cornerIds;
};

template< typename MeshConfig >
class EntitySeed< MeshConfig, Topologies::Polygon >
{
   using MeshConfigTraits = MeshTraits< MeshConfig >;

public:
   using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
   using LocalIndexType  = typename MeshTraits< MeshConfig >::LocalIndexType;
   using DeviceType      = typename MeshTraits< MeshConfig >::DeviceType;
   using IdArrayType     = Containers::Array< GlobalIndexType, DeviceType, LocalIndexType >;
   using HashType        = EntitySeedHash< EntitySeed >;
   using KeyEqual        = EntitySeedEq< EntitySeed >;

   void setCornersCount( const LocalIndexType& cornersCount )
   {
      TNL_ASSERT_GE( cornersCount, 3, "cornersCount must be at least 3" );
      this->cornerIds.setSize( cornersCount );
   }

   LocalIndexType getCornersCount() const
   {
      return this->cornerIds.getSize();
   }

   void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
   {
      TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
      TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
      TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

      this->cornerIds[ cornerIndex ] = pointIndex;
   }

   IdArrayType& getCornerIds()
   {
      return cornerIds;
   }

   const IdArrayType& getCornerIds() const
   {
      return cornerIds;
   }

private:
   IdArrayType cornerIds;
};

template< typename MeshConfig, typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const EntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

template< typename EntitySeed >
struct EntitySeedHash
{
   std::size_t operator()( const EntitySeed& seed ) const
   {
      using LocalIndexType = typename EntitySeed::LocalIndexType;
      using GlobalIndexType = typename EntitySeed::GlobalIndexType;

      // Note that we must use an associative function to combine the hashes,
      // because we *want* to ignore the order of the corner IDs.
      std::size_t hash = 0;
      for( LocalIndexType i = 0; i < seed.getCornersCount(); i++ )
//         hash ^= std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
         hash += std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
      return hash;
   }
};

template< typename EntitySeed >
struct EntitySeedEq
{
   bool operator()( const EntitySeed& left, const EntitySeed& right ) const
   {
      using IdArrayType = typename EntitySeed::IdArrayType;

      IdArrayType sortedLeft( left.getCornerIds() );
      IdArrayType sortedRight( right.getCornerIds() );
      sortedLeft.sort();
      sortedRight.sort();
      return sortedLeft == sortedRight;
   }
};

template< typename MeshConfig >
struct EntitySeedEq< EntitySeed< MeshConfig, Topologies::Vertex > >
{
   using Seed = EntitySeed< MeshConfig, Topologies::Vertex >;

   bool operator()( const Seed& left, const Seed& right ) const
   {
      return left.getCornerIds() == right.getCornerIds();
   }
};

} // namespace Meshes
} // namespace TNL
