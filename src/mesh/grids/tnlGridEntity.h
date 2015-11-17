/***************************************************************************
                          tnlGridEntity.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLGRIDENTITY_H
#define	TNLGRIDENTITY_H

template< typename Grid,
          int EntityDimensions >
class tnlGridEntity
{
   public:
      
      typedef Grid GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = EntityDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityProportionsType;
      
      tnlGridEntity();
      
      tnlGridEntity( const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityProportionsType& proportions );
      
      const CoordinatesType& getCoordinates() const;
      
      CoordinatesType& getCoordinates();
      
      const EntityOrientationType& getOrientation() const;

   protected:
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityProportionsType proportions;
};

#endif	/* TNLGRIDENTITY_H */

