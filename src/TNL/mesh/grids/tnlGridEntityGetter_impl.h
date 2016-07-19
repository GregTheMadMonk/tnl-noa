/***************************************************************************
                          tnlGridEntityGetter_impl.h  -  description
                             -------------------
    begin                : Nov 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/grids/tnlGridEntityGetter.h>
#include <TNL/mesh/grids/tnlGrid1D.h>
#include <TNL/mesh/grids/tnlGrid2D.h>
#include <TNL/mesh/grids/tnlGrid3D.h>

namespace TNL {

/****
 * 1D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity,
          int EntityDimensions >
class tnlGridEntityGetter<
   tnlGrid< 1, Real, Device, Index >,
   GridEntity,
   EntityDimensions >
{
   public:
 
      static const int entityDimensions = EntityDimensions;
 
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
              std::cerr << " index = " << index
                   << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                   << " entityDimensions = " << entityDimensions );
         return GridEntity
            ( grid,
              CoordinatesType( index ),
              typename GridEntity::EntityOrientationType( 0 ),
              typename GridEntity::EntityBasisType( EntityDimensions ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + CoordinatesType( 1 - entityDimensions ),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return entity.getCoordinates().x();
      }
};

/****
 * 2D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
 
      static const int entityDimensions = 2;
 
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();

         return GridEntity
            ( grid,
              CoordinatesType( index % dimensions.x(),
                               index / dimensions.x() ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 1, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions() );

         //const CoordinatesType coordinates = entity.getCoordinates();
         //const CoordinatesType dimensions = grid.getDimensions();
 
         return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
      }
 
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
 
      static const int entityDimensions = 1;
 
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions, EntityConfig > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );
 
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % aux, index / aux ),
                 typename GridEntity::EntityOrientationType( 1, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxFaces;
         const IndexType& aux = dimensions.x();
         return GridEntity
            ( grid,
              CoordinatesType( i % aux, i / aux ),
              typename GridEntity::EntityOrientationType( 0, 1 ),
              typename GridEntity::EntityBasisType( 1, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + abs( entity.getOrientation() ),
                 std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                      << " dimensions.x() = " << grid.getDimensions()
                      << " abs( entity.getOrientation() ) = " << abs( entity.getOrientation() ) );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         if( entity.getOrientation().x() )
            return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
         return grid.numberOfNxFaces + coordinates.y() * dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
 
      static const int entityDimensions = 0;
 
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();

         const IndexType aux = dimensions.x() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( index % aux,
                               index / aux ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= 0 && entity.getCoordinates() <= grid.getDimensions(),
            std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                 << " grid.getDimensions() = " << grid.getDimensions() );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
      }
};

/****
 * 3D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, GridEntity, 3 >
{
   public:
 
      static const int entityDimensions = 3;
 
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();

         return GridEntity
            ( grid,
              CoordinatesType( index % dimensions.x(),
                               ( index / dimensions.x() ) % dimensions.y(),
                               index / ( dimensions.x() * dimensions.y() ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 1, 1, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions() );

         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return ( coordinates.z() * dimensions.y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, GridEntity, 2 >
{
   public:
 
      static const int entityDimensions = 2;
 
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();
 
         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % aux,
                                  ( index / aux ) % dimensions.y(),
                                  index / ( aux * dimensions.y() ) ),
                 typename GridEntity::EntityOrientationType( 1, 0, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1, 1 ) );
         }
         if( index < grid.numberOfNxAndNyFaces )
         {
            const IndexType i = index - grid.numberOfNxFaces;
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( i % dimensions.x(),
                                  ( i / dimensions.x() ) % aux,
                                  i / ( aux * dimensions.x() ) ),
                 typename GridEntity::EntityOrientationType( 0, 1, 0 ),
                 typename GridEntity::EntityBasisType( 1, 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxAndNyFaces;
         return GridEntity
            ( grid,
              CoordinatesType( i % dimensions.x(),
                               ( i / dimensions.x() ) % dimensions.y(),
                               i / ( dimensions.x() * dimensions.y() ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 1 ),
              typename GridEntity::EntityBasisType( 1, 1, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + abs( entity.getOrientation() ),
                 std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                      << " dimensions.x() = " << grid.getDimensions()
                      << " abs( entity.getOrientation() ) = " << abs( entity.getOrientation() ) );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();

 
         if( entity.getOrientation().x() )
         {
            return ( coordinates.z() * dimensions.y() + coordinates.y() ) *
               ( dimensions.x() + 1 ) + coordinates.x();
         }
         if( entity.getOrientation().y() )
         {
            return grid.numberOfNxFaces +
               ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) *
               dimensions.x() + coordinates.x();
         }
         return grid.numberOfNxAndNyFaces +
            ( coordinates.z() * dimensions.y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, GridEntity, 1 >
{
   public:
 
      static const int entityDimensions = 1;
 
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );
 
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfDxEdges )
         {
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % dimensions.x(),
                                  ( index / dimensions.x() ) % aux,
                                  index / ( dimensions.x() * aux ) ),
                 typename GridEntity::EntityOrientationType( 0, 0, 0 ),
                 typename GridEntity::EntityBasisType( 1, 0, 0 ) );

         }
         if( index < grid.numberOfDxAndDyEdges )
         {
            const IndexType i = index - grid.numberOfDxEdges;
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( i % aux,
                                  ( i / aux ) % dimensions.y(),
                                  i / ( aux * dimensions.y() ) ),
                 typename GridEntity::EntityOrientationType( 0, 0, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1, 0 ) );
         }
         const IndexType i = index - grid.numberOfDxAndDyEdges;
         const IndexType aux1 = dimensions.x() + 1;
         const IndexType aux2 = dimensions.y() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( i % aux1,
                               ( i / aux1 ) % aux2,
                               i / ( aux1 * aux2 ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() +
                       CoordinatesType( 1, 1, 1 ) - entity.getBasis(),
            std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                 << " dimensions.x() = " << grid.getDimensions()
                 << " CoordinatesType( 1, 1, 1 ) - entity.getBasis() = " << CoordinatesType( 1, 1, 1 ) - entity.getBasis() );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         if( entity.getBasis().x() )
            return ( coordinates.z() * ( dimensions.y() + 1 ) +
                     coordinates.y() ) * dimensions.x() + coordinates.x();
         if( entity.getBasis().y() )
            return grid.numberOfDxEdges +
               ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) +
               coordinates.x();
         return grid.numberOfDxAndDyEdges +
            ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) +
            coordinates.x();

      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, GridEntity, 0 >
{
   public:
 
      static const int entityDimensions = 0;
 
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< GridEntity >(),
           std::cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< GridEntity >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();
 
         const IndexType auxX = dimensions.x() + 1;
         const IndexType auxY = dimensions.y() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( index % auxX,
                               ( index / auxX ) % auxY,
                               index / ( auxX * auxY ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= 0 && entity.getCoordinates() <= grid.getDimensions(),
            std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                 << " grid.getDimensions() = " << grid.getDimensions() );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) *
                ( dimensions.x() + 1 ) +
                coordinates.x();
      }
};

} // namespace TNL

