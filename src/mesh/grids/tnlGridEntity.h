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

#include <mesh/grids/tnlNeighbourGridEntitiesStorage.h>

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter;

template< typename GridEntityType >
class tnlBoundaryGridEntityChecker;

template< typename GridEntityType >
class tnlGridEntityCenterGetter;


template< typename Grid,          
          int EntityDimensions,
          typename Config >
class tnlGridEntity
{
};


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,          
          int EntityDimensions,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = EntityDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions, Config > ThisType;
      typedef typename GridType::VertexType VertexType;
      
      typedef tnlNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlNeighbourGridEntityGetter<
            tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           EntityDimensions,
                           Config >,
            NeighbourEntityDimensions >;
      
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

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh();

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
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;
      
      __cuda_callable__ inline
      const GridType& getMesh() const;
      
   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
      //__cuda_callable__ inline
      //tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
};

/****
 * Specializations for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = meshDimensions;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions, Config > ThisType;
      typedef tnlNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlNeighbourGridEntityGetter<
            tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;


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

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh();

      __cuda_callable__ inline
      Index getIndex() const;
            
      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const;     
      
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = Dimensions >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;
      
      __cuda_callable__ inline
      const VertexType& getEntityProportions() const;      
      
      __cuda_callable__ inline
      const GridType& getMesh() const;

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
      //__cuda_callable__ inline
      //tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
};

/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;      
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = 0;
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions, Config > ThisType;
      typedef tnlNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlNeighbourGridEntityGetter<
            tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;


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

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh();

      __cuda_callable__ inline
      Index getIndex() const;
            
      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const;     
      
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const;
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
      
      __cuda_callable__ inline
      VertexType getCenter() const;
      
      __cuda_callable__ inline
      VertexType getEntityProportions() const;
      
      __cuda_callable__ inline
      const GridType& getMesh() const;
      
   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
      //__cuda_callable__ inline
      //tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
};

#include <mesh/grids/tnlGridEntity_impl.h>

#endif	/* TNLGRIDENTITY_H */

