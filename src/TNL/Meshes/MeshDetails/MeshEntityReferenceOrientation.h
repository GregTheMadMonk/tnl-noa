/***************************************************************************
                          MeshEntityReferenceOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
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

#include <map>

#include <TNL/Meshes/MeshDetails/MeshEntityOrientation.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology >
class MeshEntityReferenceOrientation
{
	typedef typename MeshTraits< MeshConfig >::LocalIndexType  LocalIndexType;
	typedef typename MeshTraits< MeshConfig >::GlobalIndexType GlobalIndexType;

   public:
      typedef EntitySeed< MeshConfig, EntityTopology >            SeedType;
      typedef MeshEntityOrientation< MeshConfig, EntityTopology > EntityOrientation;

      MeshEntityReferenceOrientation() = default;

      explicit MeshEntityReferenceOrientation( const SeedType& referenceSeed )
      {
         auto referenceCornerIds = referenceSeed.getCornerIds();
         for( LocalIndexType i = 0; i < referenceCornerIds.getSize(); i++ )
         {
            TNL_ASSERT_TRUE( this->cornerIdsMap.find( referenceCornerIds[ i ] ) == this->cornerIdsMap.end(),
                             "detected duplicate index in the reference seed" );
            this->cornerIdsMap.insert( std::make_pair( referenceCornerIds[i], i ) );
         }
      }
 
      static String getType(){ return "MeshEntityReferenceOrientation"; };

      EntityOrientation createOrientation( const SeedType& seed ) const
      {
         EntityOrientation result;
         auto cornerIds = seed.getCornerIds();
         for( LocalIndexType i = 0; i < cornerIds.getSize(); i++ )
         {
            TNL_ASSERT_TRUE( this->cornerIdsMap.find( cornerIds[ i ] ) != this->cornerIdsMap.end(),
                             "unable to find index for entity orientation" );
            result.setPermutationValue( i, this->cornerIdsMap.find( cornerIds[ i ])->second );
         }
         return result;
      }

   private:
      std::map< GlobalIndexType, LocalIndexType > cornerIdsMap;
};

} // namespace Meshes
} // namespace TNL

