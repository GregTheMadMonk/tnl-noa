/***************************************************************************
                          tnlGridEntityGetter_impl.h  -  description
                             -------------------
    begin                : Nov 15, 2015
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

#ifndef TNLGRIDENTITYINDEXER_IMPL_H
#define	TNLGRIDENTITYINDEXER_IMPL_H

#include <mesh/grids/tnlGridEntityGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>


/****
 * 1D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
class tnlGridEntityGetter< tnlGrid< 1, Real, Device, Index >,
                            EntityDimensions >
{
   public:
      
      static const int entityDimensions = EntityDimensions;
      
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {         
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
              cerr << " index = " << index
                   << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                   << " entityDimensions = " << entityDimensions );
         return GridEntity
            ( CoordinatesType( index ),
              typename GridType::EntityOrientation( 0 ),
              typename GridType::EntityProportions( entityDimensions ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + CoordinatesType( 1 - entityDimensions ),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
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
          typename Index >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, 2 >
{
   public:
      
      static const int entityDimensions = 2;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();         

         return GridEntity
            ( CoordinatesType( index % dimensions.x(),
                               index / dimensions.x() ),
              typename GridType::EntityOrientation( 0, 0 ),
              typename GridType::EntityProportions( 1, 1 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions() );

         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         return coordinates.y() * grid.getDimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, 1 >
{
   public:
      
      static const int entityDimensions = 1;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );
         
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( CoordinatesType( index % aux, index / aux ),
                 typename GridType::EntityOrientation( 1, 0 ),
                 typename GridType::EntityProportions( 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxFaces;
         const IndexType& aux = dimensions.x();
         return GridEntity
            ( CoordinatesType( i % aux, i / aux ),
              typename GridType::EntityOrientation( 0, 1 ),
              typename GridType::EntityProportions( 1, 0 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + tnlAbs( entity.getOrientation() ),
                 cerr << "entity.getCoordinates() = " << entity.getCoordinates
                      << " dimensions.x() = " << grid.getDimensions()
                      << " tnlAbs( entity.getOrientation() ) = " << tnlAbs( entity.getOrientation() ) );
                  
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         if( entity.getOrientation().x() )
            return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
         return grid.numberOfNxFaces + coordinates.y() * dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, 0 >
{
   public:
      
      static const int entityDimensions = 0;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();

         const IndexType aux = dimensions.x() + 1;
         return GridEntity
            ( CoordinatesType( index % aux, 
                               index / aux ),
              typename GridType::EntityOrientation( 0, 0 ),
              typename GridType::EntityProportions( 0, 0 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= 0 && entity.getCoordinates() <= grid.getDimensions(),
            cerr << "entity.getCoordinates() = " << entity.getCoordinates()
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
          typename Index >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, 3 >
{
   public:
      
      static const int entityDimensions = 3;
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();

         return GridEntity
            ( CoordinatesType( index % dimensions.x(),
                               ( index / dimensions.x() ) % dimensions.y(),
                               index / ( dimensions.x() * dimensions.y() ) ),
              typename GridType::EntityOrientation( 0, 0, 0 ),
              typename GridType::EntityProportions( 1, 1, 1 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions() );

         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         return ( coordinates.z() * dimensions.y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, 2 >
{
   public:
      
      static const int entityDimensions = 2;
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();
         
         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( CoordinatesType( index % aux,
                                  ( index / aux ) % dimensions.y(),
                                  index / ( aux * dimensions.y() ) ),
                 typename GridType::EntityOrientation( 1, 0, 0 ),
                 typename GridType::EntityProportions( 0, 1, 1 ) );
         }
         if( index < grid.numberOfNxAndNyFaces )
         {
            const IndexType i = index - grid.numberOfNxFaces;
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( CoordinatesType( i % dimensions.x(),
                                  ( i / dimensions.x() ) % aux,
                                  i / ( aux * dimensions.x() ) ),
                 typename GridType::EntityOrientation( 0, 1, 0 ),
                 typename GridType::EntityProportions( 1, 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxAndNyFaces;
         return GridEntity
            ( CoordinatesType( i % dimensions.x(),
                               ( i / dimensions.x() ) % dimensions.y(),
                               i / ( dimensions.x() * dimensions.y() ) ),
              typename GridType::EntityOrientation( 0, 0, 1 ),
              typename GridType::EntityProportions( 1, 1, 0 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + tnlAbs( entity.getOrientation() ),
                 cerr << "entity.getCoordinates() = " << entity.getCoordinates
                      << " dimensions.x() = " << grid.getDimensions()
                      << " tnlAbs( entity.getOrientation() ) = " << tnlAbs( entity.getOrientation() ) );
         
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();

         
         if( entity.getOrientation().x() )
         {
            return ( coordinates.z() * dimensions.y() + coordinates().y() ) * 
               ( dimensions.x() + 1 ) + coordinates.x();
         }
         if( entity.getOrientation().y() )
         {
            return grid.numberOfNxFaces + 
               ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * 
               dimensions.x() + coordinates.x();
         }
         return grid.numberOfNxAndNyFaces + 
            ( coordinates.z() * dimensions().y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, 1 >
{
   public:
      
      static const int entityDimensions = 1;
      
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );
         
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfDxEdges )
         {
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( CoordinatesType( index % dimensions.x(),
                                  ( index / dimensions.x() ) % aux,
                                  index / ( dimensions.x() * aux ) ),
                 typename GridType::EntityOrientation( 0, 0, 0 ),
                 typename GridType::EntityProportions( 1, 0, 0 ) );

         }
         if( index < grid.numberOfDxAndDyEdges )
         {
            const IndexType i = index - grid.numberOfDxEdges;
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( CoordinatesType( i % aux,
                                  ( i / aux ) % dimensions.y(),
                                  i / ( aux * dimensions.y() ) ),
                 typename GridType::EntityOrientation( 0, 0, 0 ),
                 typename GridType::EntityProportions( 0, 1, 0 ) );
         }
         const IndexType i = index - grid.numberOfDxAndDyEdges;
         const IndexType aux1 = dimensions.x() + 1;
         const IndexType aux2 = dimensions.y() + 1;
         return CoordinatesType(  );
         return GridEntity
            ( CoordinatesType( i % aux1,
                               ( i / aux1 ) % aux2,
                               i / ( aux1 * aux2 ) ),
              typename GridType::EntityOrientation( 0, 0, 0 ),
              typename GridType::EntityProportions( 0, 0, 1 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + 
                       CoordinatesType( 1, 1, 1 ) - entity.getProportions(),
            cerr << "entity.getCoordinates() = " << entity.getCoordinates
                 << " dimensions.x() = " << grid.getDimensions()
                 << " CoordinatesType( 1, 1, 1 ) - entity.getProportions() = " << CoordinatesType( 1, 1, 1 ) - entity.getProportions() );
         
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         if( entity.getProportions().x() )
            return ( coordinates.z() * ( dimensions.y() + 1 ) + 
                     coordinates.y() ) * dimensions.x() + coordinates.x();   
         if( entity.getProportions().y() )
            return grid.numberOfDxEdges + 
               ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) +
               coordinates.x();
         return grid.numberOfDxAndDyEdges + 
            ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions().x() + 1 ) +
            coordinates.x();

      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityGetter< tnlGrid< 3, Real, Device, Index >, 0 >
{
   public:
      
      static const int entityDimensions = 0;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         tnlAssert( index >= 0 && index < grid.template getEntitiesCount< entityDimensions >(),
           cerr << " index = " << index
                << " grid.getEntitiesCount<>() = " << grid.template getEntitiesCount< entityDimensions >()
                << " entityDimensions = " << entityDimensions );

         const CoordinatesType dimensions = grid.getDimensions();
         
         const IndexType auxX = dimensions.x() + 1;
         const IndexType auxY = dimensions.y() + 1;
         return GridEntity
            ( CoordinatesType( index % auxX,
                               ( index / auxX ) % auxY,
                               index / ( auxX * auxY ) ),
              typename GridType::EntityOrientation( 0, 0, 0 ),
              typename GridType::EntityProportions( 0, 0, 0 ) );
      }
      
      __cuda_callable__
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         tnlAssert( entity.getCoordinates() >= 0 && entity.getCoordinates() <= grid.getDimensions(),
            cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                 << " grid.getDimensions() = " << grid.getDimensions() );
         
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * 
                ( dimensions.x() + 1 ) + 
                coordinates.x();
      }
};


#endif	/* TNLGRIDENTITYINDEXER_IMPL_H */

