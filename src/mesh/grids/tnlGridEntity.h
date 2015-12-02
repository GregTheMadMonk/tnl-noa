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

template< typename GridEntityType >
class tnlBoundaryGridEntityChecker;

template< typename GridEntityType >
class tnlGridEntityCenterGetter;


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
      typedef typename GridType::VertexType VertexType;
      
      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid );
      
      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityBasisType& basis );
      
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const;
      
      __cuda_callable__ inline
      CoordinatesType& getCoordinates();
      
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates );

      __cuda_callable__ inline
      void setIndex( IndexType entityIndex );

      __cuda_callable__ inline
      Index getIndex() const;
      
      __cuda_callable__ inline
      const EntityOrientationType& getOrientation() const;
      
      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation );
      
      __cuda_callable__ inline
      const EntityBasisType& getBasis() const;
      
      __cuda_callable__ inline
      EntityBasisType& getBasis();
      
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis );
      
      template< int NeighbourEntityDimensions = entityDimensions >
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter<
         tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                        EntityDimensions >,
         NeighbourEntityDimensions >
      getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
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
      typedef typename GridType::VertexType VertexType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = meshDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions > ThisType;

      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid );
      
      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) );
      
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const;      
      
      __cuda_callable__ inline
      CoordinatesType& getCoordinates();  
      
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates );

      __cuda_callable__ inline
      void setIndex( IndexType entityIndex );

      __cuda_callable__ inline
      Index getIndex() const;
            
      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const;     
      
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = Dimensions >
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter<
         tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >,
         NeighbourEntityDimensions >
      getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
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
      typedef typename GridType::VertexType VertexType;
      
      static const int meshDimensions = GridType::Dimensions;
      
      static const int entityDimensions = 0;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions > ThisType;

      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid );
      
      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation,
                     const EntityBasisType& basis );
      
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const;
      
      __cuda_callable__ inline
      CoordinatesType& getCoordinates();
      
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates );
      
      __cuda_callable__ inline
      void setIndex( IndexType entityIndex ) const;

      __cuda_callable__ inline
      Index getIndex() const;
            
      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const;     
      
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter< ThisType, NeighbourEntityDimensions > getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
};

#include <mesh/grids/tnlGridEntity_impl.h>

#endif	/* TNLGRIDENTITY_H */

