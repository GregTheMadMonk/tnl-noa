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
          int NeighbourEntityDimensions,
          typename StencilStorage >
class NeighbourGridEntityGetter;

template< typename GridEntityType >
class BoundaryGridEntityChecker;

template< typename GridEntityType >
class GridEntityCenterGetter;


template< typename Grid,
          int EntityDimensions,
          typename Config >
class GridEntity
{
};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
class GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >
{
   public:
 
      typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;
 
      static const int meshDimensions = GridType::meshDimensions;
 
      static const int entityDimensions = EntityDimensions;
 
      constexpr static int getDimensions() { return EntityDimensions; };
 
      constexpr static int getMeshDimensions() { return meshDimensions; };
 
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimensions, Config > ThisType;
      typedef typename GridType::VertexType VertexType;
 
      typedef NeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >,
                           EntityDimensions,
                           Config >,
            NeighbourEntityDimensions >;
 
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
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >
{
   public:
 
      typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
 
      static const int meshDimensions = GridType::meshDimensions;
 
      static const int entityDimensions = meshDimensions;

      constexpr static int getDimensions() { return entityDimensions; };
 
      constexpr static int getMeshDimensions() { return meshDimensions; };
 
 
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimensions, Config > ThisType;
      typedef NeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;


      __cuda_callable__ inline
      GridEntity( const GridType& grid );
 
      __cuda_callable__ inline
      GridEntity( const GridType& grid,
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
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};
 
      template< int NeighbourEntityDimensions = Dimensions >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const;
 
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      VertexType getCenter() const;
 
      __cuda_callable__ inline
      const RealType& getMeasure() const;
 
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
      //GridEntity();
 
      friend class BoundaryGridEntityChecker< ThisType >;
 
      friend class GridEntityCenterGetter< ThisType >;
};

/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >
{
   public:
 
      typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
 
      static const int meshDimensions = GridType::meshDimensions;
 
      static const int entityDimensions = 0;
 
      constexpr static int getDimensions() { return entityDimensions; };
 
      constexpr static int getMeshDimensions() { return meshDimensions; };
 
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef GridEntity< GridType, entityDimensions, Config > ThisType;
      typedef NeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
 
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities =
         NeighbourGridEntityGetter<
            GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;


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
      const EntityOrientationType getOrientation() const;
 
      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};

 
      template< int NeighbourEntityDimensions = entityDimensions >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const;
 
      __cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      VertexType getCenter() const;

      __cuda_callable__ inline
      const RealType getMeasure() const;
 
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
 
      friend class BoundaryGridEntityChecker< ThisType >;
 
      friend class GridEntityCenterGetter< ThisType >;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/GridEntity_impl.h>

