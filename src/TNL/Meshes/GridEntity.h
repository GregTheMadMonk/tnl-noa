/***************************************************************************
                          GridEntity.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/NeighbourGridEntitiesStorage.h>

namespace TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighbourEntityDimension,
          typename StencilStorage >
class NeighbourGridEntityGetter;

template< typename GridEntityType >
class BoundaryGridEntityChecker;

template< typename GridEntityType >
class GridEntityCenterGetter;


template< typename Grid,
          int EntityDimension,
          typename Config >
class GridEntity
{
};

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >
{
   public:
 
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;
 
      static const int meshDimension = GridType::meshDimension;
 
      static const int entityDimension = EntityDimension;
 
      constexpr static int getDimension() { return EntityDimension; };
 
      constexpr static int getMeshDimension() { return meshDimension; };            
 
      typedef Containers::StaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef Containers::StaticVector< meshDimension, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimension, Config > ThisType;
      typedef typename GridType::PointType PointType;
 
      typedef NeighbourGridEntitiesStorage< ThisType, Config > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimension = entityDimension >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           EntityDimension,
                           Config >,
            NeighbourEntityDimension >;
 
      __cuda_callable__ inline
      GridEntity( const GridType& grid );
 
      __cuda_callable__ inline
      GridEntity( const GridType& grid,
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
      void setBasis( const EntityBasisType& basis );
 
      template< int NeighbourEntityDimension = entityDimension >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimension >&
      getNeighbourEntities() const;
 
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      PointType getCenter() const;
 
      __cuda_callable__ inline
      const RealType& getMeasure() const;
 
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
      //GridEntity();
 
      friend class BoundaryGridEntityChecker< ThisType >;
 
      friend class GridEntityCenterGetter< ThisType >;
};

/****
 * Specializations for cells
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >
{
   public:
 
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      typedef Config ConfigType;
 
      static const int meshDimension = GridType::meshDimension;
 
      static const int entityDimension = meshDimension;

      constexpr static int getDimension() { return entityDimension; };
 
      constexpr static int getMeshDimension() { return meshDimension; };
 
 
      typedef Containers::StaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef Containers::StaticVector< meshDimension, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimension, Config > ThisType;
      typedef NeighbourGridEntitiesStorage< ThisType, Config > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimension = entityDimension >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           entityDimension,
                           Config >,
            NeighbourEntityDimension >;


      __cuda_callable__ inline
      GridEntity( const GridType& grid );
 
      __cuda_callable__ inline
      GridEntity( const GridType& grid,
                  const CoordinatesType& coordinates,
                  const EntityOrientationType& orientation = EntityOrientationType( ( Index ) 0 ),
                  const EntityBasisType& basis = EntityBasisType( ( Index ) 1 ) );
 
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
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};
 
      template< int NeighbourEntityDimension = Dimension >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimension >&
      getNeighbourEntities() const;
 
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      PointType getCenter() const;
 
      __cuda_callable__ inline
      const RealType& getMeasure() const;
 
      __cuda_callable__ inline
      const PointType& getEntityProportions() const;
 
      __cuda_callable__ inline
      const GridType& getMesh() const;

   protected:
 
      const GridType& grid;
 
      IndexType entityIndex;
 
      CoordinatesType coordinates;
 
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
 
      //__cuda_callable__ inline
      //GridEntity();
 
      friend class BoundaryGridEntityChecker< ThisType >;
 
      friend class GridEntityCenterGetter< ThisType >;
};

/****
 * Specialization for vertices
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >
{
   public:
 
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      typedef Config ConfigType;
 
      static const int meshDimension = GridType::meshDimension;
 
      static const int entityDimension = 0;
 
      constexpr static int getDimension() { return entityDimension; };
 
      constexpr static int getMeshDimension() { return meshDimension; };
 
      typedef Containers::StaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef Containers::StaticVector< meshDimension, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimension, Config > ThisType;
      typedef NeighbourGridEntitiesStorage< ThisType, Config > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimension = entityDimension >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           entityDimension,
                           Config >,
            NeighbourEntityDimension >;


      __cuda_callable__ inline
      GridEntity( const GridType& grid );
 
      __cuda_callable__ inline
      GridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( ( Index ) 0 ),
                     const EntityBasisType& basis = EntityOrientationType( ( Index ) 0 ) );
 
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
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};

 
      template< int NeighbourEntityDimension = entityDimension >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimension >&
      getNeighbourEntities() const;
 
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      PointType getCenter() const;

      __cuda_callable__ inline
      const RealType getMeasure() const;
 
      __cuda_callable__ inline
      PointType getEntityProportions() const;
 
      __cuda_callable__ inline
      const GridType& getMesh() const;
 
   protected:
 
      const GridType& grid;
 
      IndexType entityIndex;
 
      CoordinatesType coordinates;
 
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
 
      friend class BoundaryGridEntityChecker< ThisType >;
 
      friend class GridEntityCenterGetter< ThisType >;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/GridEntity_impl.h>

