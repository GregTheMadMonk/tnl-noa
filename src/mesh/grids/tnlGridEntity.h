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
};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = EntityDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      
      tnlGridEntity();
      
      tnlGridEntity( const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityBasisType& basis );
      
      const CoordinatesType& getCoordinates() const;
      
      CoordinatesType& getCoordinates();
      
      void setCoordinates( const CoordinatesType& coordinates );
      
      const EntityOrientationType& getOrientation() const;
      
      void setOrientation( const EntityOrientationType& orientation );
      
      const EntityBasisType& getBasis() const;
      
      EntityBasisType& getBasis();
      
      void setBasis( const EntityBasisType& basis );

   protected:
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
};

/****
 * Specializations for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = meshDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;

      tnlGridEntity();
      
      tnlGridEntity( const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) );
      
      const CoordinatesType& getCoordinates() const;      
      
      CoordinatesType& getCoordinates();  
      
      void setCoordinates( const CoordinatesType& coordinates );
      
      const EntityOrientationType getOrientation() const;     
      
      const EntityBasisType getBasis() const;
            
   protected:
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      EntityBasisType basis;
};

/****
 * Specialization for vertices
 */

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = 0;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;

      tnlGridEntity();
      
      tnlGridEntity( const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityBasisType& basis );
      
      const CoordinatesType& getCoordinates() const;
      
      CoordinatesType& getCoordinates();
      
      void setCoordinates( const CoordinatesType& coordinates );
      
      const EntityOrientationType getOrientation() const;     
      
      const EntityBasisType getBasis() const;
            
   protected:
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;

};

#include <mesh/grids/tnlGridEntity_impl.h>

#endif	/* TNLGRIDENTITY_H */

