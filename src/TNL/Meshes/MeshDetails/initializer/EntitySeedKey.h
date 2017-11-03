/***************************************************************************
                          EntitySeedKey.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeed;

/****
 * Unique identification of a mesh entity by its vertices.
 * Uniqueness is preserved for entities of the same type only.
 */
template< typename MeshConfig,
          typename EntityTopology >
class EntitySeedKey
{
   using EntitySeedType = EntitySeed< MeshConfig, EntityTopology >;
   using IdArrayType = typename EntitySeedType::IdArrayType;

public:
   EntitySeedKey( const EntitySeedType& entitySeed )
   {
      for( typename IdArrayType::IndexType i = 0;
           i < entitySeed.getCornersCount();
           i++ )
         this->sortedCorners[ i ] = entitySeed.getCornerIds()[ i ];
      sortedCorners.sort( );
   }

   bool operator<( const EntitySeedKey& other ) const
   {
      for( typename IdArrayType::IndexType i = 0;
           i < IdArrayType::size;
           i++)
      {
         if( sortedCorners[ i ] < other.sortedCorners[ i ] )
            return true;
         else
            if( sortedCorners[ i ] > other.sortedCorners[ i ] )
               return false;
      }
      return false;
   }

private:
   IdArrayType sortedCorners;
};

} // namespace Meshes
} // namespace TNL
