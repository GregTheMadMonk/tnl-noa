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

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntitySeed;

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSubentitiesTraits;

/****
 * Unique identification of a mesh entity by its vertices.
 * Uniqueness is preserved for entities of the same type only.
 */
template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntitySeedKey
{
   typedef
      tnlMeshEntitySeed< ConfigTag, EntityTag >                               EntitySeedType;

   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                EntityTag,
                                tnlDimensionsTag< 0 > >::ContainerType ContainerType;

   public:

   explicit tnlMeshEntitySeedKey( const EntitySeedType& entitySeed )
   {
      for( typename ContainerType::IndexType i = 0; 
           i < entitySeed.getCornersCount();
           i++ )
         this->sortedCorners[ i ] = entitySeed.getCornerIds()[ i ];
      sortedCorners.sort( );
   }

   bool operator<( const tnlMeshEntitySeedKey& other ) const
   {
      for( typename ContainerType::IndexType i = 0;
           i < ContainerType::size;
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

   ContainerType sortedCorners;
};


#endif /* TNLMESHENTITYKSEEDEY_H_ */
