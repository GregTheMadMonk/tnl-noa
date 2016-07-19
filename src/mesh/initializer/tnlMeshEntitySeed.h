/***************************************************************************
                          tnlMeshEntitySeed.h  -  description
                             -------------------
    begin                : Aug 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/traits/tnlMeshTraits.h>

namespace TNL {

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntitySeed
{
   typedef tnlMeshTraits< MeshConfig >      MeshConfigTraits;
   typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 > SubvertexTraits;

   public:
      typedef typename tnlMeshTraits< MeshConfig >::GlobalIndexType                                      GlobalIndexType;
      typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType                                       LocalIndexType;
      typedef typename tnlMeshTraits< MeshConfig >::IdArrayAccessorType                                  IdArrayAccessorType;
      typedef typename SubvertexTraits::IdArrayType                                                      IdArrayType;

      static tnlString getType() { return tnlString( "tnlMeshEntitySeed<>" ); }
 
      static constexpr LocalIndexType getCornersCount()
      {
         return SubvertexTraits::count;
      }

      void setCornerId( LocalIndexType cornerIndex, GlobalIndexType pointIndex )
      {
         tnlAssert( 0 <= cornerIndex && cornerIndex < getCornersCount(), std::cerr << "cornerIndex = " << cornerIndex );
         tnlAssert( 0 <= pointIndex, std::cerr << "pointIndex = " << pointIndex );

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
std::ostream& operator << ( std::ostream& str, const tnlMeshEntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

} // namespace TNL

