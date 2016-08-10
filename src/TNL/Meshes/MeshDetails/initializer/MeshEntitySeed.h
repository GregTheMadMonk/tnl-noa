/***************************************************************************
                          MeshEntitySeed.h  -  description
                             -------------------
    begin                : Aug 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntitySeed
{
   typedef MeshTraits< MeshConfig >      MeshConfigTraits;
   typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 > SubvertexTraits;

   public:
      typedef typename MeshTraits< MeshConfig >::GlobalIndexType                                      GlobalIndexType;
      typedef typename MeshTraits< MeshConfig >::LocalIndexType                                       LocalIndexType;
      typedef typename MeshTraits< MeshConfig >::IdArrayAccessorType                                  IdArrayAccessorType;
      typedef typename SubvertexTraits::IdArrayType                                                      IdArrayType;

      static String getType() { return String( "MeshEntitySeed<>" ); }
 
      static constexpr LocalIndexType getCornersCount()
      {
         return SubvertexTraits::count;
      }

      void setCornerId( LocalIndexType cornerIndex, GlobalIndexType pointIndex )
      {
         Assert( 0 <= cornerIndex && cornerIndex < getCornersCount(), std::cerr << "cornerIndex = " << cornerIndex );
         Assert( 0 <= pointIndex, std::cerr << "pointIndex = " << pointIndex );

         this->cornerIds[ cornerIndex ] = pointIndex;
      }

      IdArrayAccessorType getCornerIds()
      {
         IdArrayAccessorType accessor;
         accessor.bind( this->corners.getData(), this->corners.getSize() );
         return accessor;
      }

 
      const IdArrayAccessorType getCornerIds() const
      {
         IdArrayAccessorType accessor;
         accessor.bind( this->cornerIds.getData(), this->cornerIds.getSize() );
         return accessor;
      }

   private:
	
      IdArrayType cornerIds;
};

template< typename MeshConfig, typename EntityTopology >
std::ostream& operator << ( std::ostream& str, const MeshEntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

} // namespace Meshes
} // namespace TNL

