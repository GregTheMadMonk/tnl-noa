/***************************************************************************
                          MeshEntitySeedKey.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDimensionsTag.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
class MeshEntitySeed;

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class MeshSubentityTraits;

/****
 * Unique identification of a mesh entity by its vertices.
 * Uniqueness is preserved for entities of the same type only.
 */
template< typename MeshConfig,
          typename EntityTopology >
class MeshEntitySeedKey
{
   typedef
      MeshEntitySeed< MeshConfig, EntityTopology >                               EntitySeedType;

   typedef typename
      MeshSubentityTraits< MeshConfig,
                                EntityTopology,
                                0 >::StorageArrayType  StorageArrayType;

   public:

   explicit MeshEntitySeedKey( const EntitySeedType& entitySeed )
   {
      for( typename StorageArrayType::IndexType i = 0;
           i < entitySeed.getCornersCount();
           i++ )
         this->sortedCorners[ i ] = entitySeed.getCornerIds()[ i ];
      sortedCorners.sort( );
   }

   bool operator<( const MeshEntitySeedKey& other ) const
   {
      for( typename StorageArrayType::IndexType i = 0;
           i < StorageArrayType::size;
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

   StorageArrayType sortedCorners;
};

} // namespace Meshes
} // namespace TNL
