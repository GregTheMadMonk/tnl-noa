/***************************************************************************
                          tnlMeshEntityKey.h  -  description
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

#ifndef TNLMESHENTITYKEY_H_
#define TNLMESHENTITYKEY_H_

#include <mesh/tnlMeshEntity.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/traits/tnlDimensionsTraits.h>

/****
 * Unique identification of a mesh entity by its vertices.
 * Uniqueness is preserved for entities of the same type only.
 */
template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityKey
{
   typedef
      tnlMeshEntity< ConfigTag, EntityTag >                               EntityType;

   typedef typename
      tnlMeshSubentitiesTraits< ConfigTag,
                                EntityTag,
                                tnlDimensionsTraits< 0 > >::ContainerType ContainerType;

   public:

   explicit tnlMeshEntityKey( const EntityType& entity )
   {
      vertexIDs.copy( entity.template subentityIndices<0>() );
      vertexIDs.sort( );
   }

   bool operator<( const tnlMeshEntityKey& other ) const
   {
      for( typename ContainerType::IndexType i = 0;
           i < vertexIDs.getSize();
           i++)
      {
         if( vertexIDs[ i ] < other.vertexIDs[ i ] )
            return true;
         else
            if( vertexIDs[ i ] > other.vertexIDs[ i ] )
               return false;
      }
      return false;
   }

   private:

   ContainerType vertexIDs;
};


#endif /* TNLMESHENTITYKEY_H_ */
