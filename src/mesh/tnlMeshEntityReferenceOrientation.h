/***************************************************************************
                          tnlMeshEntityReferenceOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
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

#ifndef TNLMESHENTITYREFERENCEORIENTATION_H
#define	TNLMESHENTITYREFERENCEORIENTATION_H

template< typename MeshConfig, typename EntityTopology >
class tnlMeshEntityReferenceOrientation
{
	typedef typename tnlMeshConfigTraits< MeshConfig >::LocalIndexType  LocalIndexType;
	typedef typename tnlMeshConfigTraits< MeshConfig >::GlobalIndexType GlobalIndexType;

   public:
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >         SeedType;
      typedef tnlMeshEntityOrientation< MeshConfig, EntityTopology >         EntityOrientation;

      tnlMeshEntityReferenceOrientation() = default;

      explicit tnlMeshEntityReferenceOrientation( const SeedType& referenceSeed )
      {
         auto referenceCornerIds = referenceSeed.getCornerIds();
         for( LocalIndexType i = 0; i < referenceCornerIds.getSize(); i++ )
         {
            tnlAssert( this->cornerIdsMap.find( referenceCornerIds[i]) == this->cornerIdsMap.end(), );
            this->cornerIdsMap.insert( std::make_pair( referenceCornerIds[i], i ) );
         }
      }
      
      static tnlString getType(){};

      EntityOrientation createOrientation( const SeedType& seed ) const
      {
         EntityOrientation result;
         auto cornerIds = seed.getCornerIds();
         for( LocalIndexType i = 0; i < cornerIds.getSize(); i++ )
         {
            tnlAssert( this->cornerIdsMap.find( cornerIds[ i ] ) != this->cornerIdsMap.end(), );
            result.setPermutationValue( i, this->cornerIdsMap.find( cornerIds[ i ])->second );
         }
         return result;
      }

   private:
      std::map< GlobalIndexType, LocalIndexType > cornerIdsMap;
};


#endif	/* TNLMESHENTITYREFERENCEORIENTATION_H */

