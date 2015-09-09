/***************************************************************************
                          tnlMeshEntitySeedKey.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLMESHENTITYSEEDKEY_H_
#define TNLMESHENTITYSEEDKEY_H_

#include <mesh/tnlDimensionsTag.h>

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntitySeed;

template< typename MeshConfig,
          typename EntityTopology,
          int Dimensions >
class tnlMeshSubentityTraits;

/****
 * Unique identification of a mesh entity by its vertices.
 * Uniqueness is preserved for entities of the same type only.
 */
template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshEntitySeedKey
{
   typedef
      tnlMeshEntitySeed< MeshConfig, EntityTopology >                               EntitySeedType;

   typedef typename
      tnlMeshSubentityTraits< MeshConfig,
                                EntityTopology,
                                0 >::StorageArrayType  StorageArrayType;

   public:

   explicit tnlMeshEntitySeedKey( const EntitySeedType& entitySeed )
   {
      for( typename StorageArrayType::IndexType i = 0; 
           i < entitySeed.getCornersCount();
           i++ )
         this->sortedCorners[ i ] = entitySeed.getCornerIds()[ i ];
      sortedCorners.sort( );
   }

   bool operator<( const tnlMeshEntitySeedKey& other ) const
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


#endif /* TNLMESHENTITYKSEEDEY_H_ */
