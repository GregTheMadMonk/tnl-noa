/***************************************************************************
                          tnlMeshEntitySeed.h  -  description
                             -------------------
    begin                : Aug 18, 2015
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

#ifndef TNLMESHENTITYSEED_H
#define	TNLMESHENTITYSEED_H

#include <mesh/traits/tnlMeshTraits.h>

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
         tnlAssert( 0 <= cornerIndex && cornerIndex < getCornersCount(), cerr << "cornerIndex = " << cornerIndex );
         tnlAssert( 0 <= pointIndex, cerr << "pointIndex = " << pointIndex );

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
ostream& operator << ( ostream& str, const tnlMeshEntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

#endif	/* TNLMESHENTITYSEED_H */

