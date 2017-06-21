/***************************************************************************
                          MeshEntitySeed.h  -  description
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
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeedKey.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntitySeed
{
   using MeshConfigTraits = MeshTraits< MeshConfig >;
   using SubvertexTraits = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 >;

   public:
      using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
      using LocalIndexType  = typename MeshTraits< MeshConfig >::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< SubvertexTraits::count, GlobalIndexType >;
      using KeyType         = MeshEntitySeedKey< MeshConfig, EntityTopology >;

      static String getType() { return String( "MeshEntitySeed<>" ); }

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
class MeshEntitySeed< MeshConfig, MeshVertexTopology >
{
   using MeshConfigTraits = MeshTraits< MeshConfig >;

   public:
      using GlobalIndexType = typename MeshTraits< MeshConfig >::GlobalIndexType;
      using LocalIndexType  = typename MeshTraits< MeshConfig >::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< 1, GlobalIndexType >;
      using KeyType         = MeshEntitySeedKey< MeshConfig, MeshVertexTopology >;

      static String getType() { return String( "MeshEntitySeed<>" ); }

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

template< typename MeshConfig, typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

} // namespace Meshes
} // namespace TNL

