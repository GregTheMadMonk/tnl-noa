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
      using IdArrayType     = typename SubvertexTraits::IdArrayType;

      static String getType() { return String( "MeshEntitySeed<>" ); }

      static constexpr LocalIndexType getCornersCount()
      {
         return SubvertexTraits::count;
      }

      void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT( 0 <= cornerIndex && cornerIndex < getCornersCount(), std::cerr << "cornerIndex = " << cornerIndex );
         TNL_ASSERT( 0 <= pointIndex, std::cerr << "pointIndex = " << pointIndex );

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

