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

template< typename GridEntity,
          int NeighbourEntityDimensions >
class tnlNeighbourGridEntityGetter;

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
      typedef tnlGridEntity< GridType, entityDimensions > ThisType;
      
      tnlGridEntity( const GridType& grid );
      
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
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
      
      template< int NeighbourEntityDimensions = entityDimensions >
      tnlNeighbourGridEntityGetter<
         tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                        EntityDimensions >,
         NeighbourEntityDimensions >
      getNeighbourEntities() const;

   protected:
      
      const GridType& grid;
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();
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
      typedef tnlGridEntity< GridType, entityDimensions > ThisType;

      tnlGridEntity( const GridType& grid );
      
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) );
      
      const CoordinatesType& getCoordinates() const;      
      
      CoordinatesType& getCoordinates();  
      
      void setCoordinates( const CoordinatesType& coordinates );
      
      const EntityOrientationType getOrientation() const;     
      
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = Dimensions >
      tnlNeighbourGridEntityGetter<
         tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >,
         NeighbourEntityDimensions >
      getNeighbourEntities() const;

            
   protected:
      
      const GridType& grid;
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();
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
      typedef tnlGridEntity< GridType, entityDimensions > ThisType;

      tnlGridEntity( const GridType& grid );
      
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityBasisType& basis );
      
      const CoordinatesType& getCoordinates() const;
      
      CoordinatesType& getCoordinates();
      
      void setCoordinates( const CoordinatesType& coordinates );
      
      const EntityOrientationType getOrientation() const;     
      
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      tnlNeighbourGridEntityGetter< ThisType, NeighbourEntityDimensions > getNeighbourEntities() const;
            
   protected:
      
      const GridType& grid;
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();

};

#include <mesh/grids/tnlGridEntity_impl.h>

#endif	/* TNLGRIDENTITY_H */

