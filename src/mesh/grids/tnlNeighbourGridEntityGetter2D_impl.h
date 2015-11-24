/***************************************************************************
                          tnlNeighbourGridEntityGetter2D_impl.h  -  description
                             -------------------
    begin                : Nov 23, 2015
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

#ifndef TNLNEIGHBOURGRIDENTITYGETTER2D_IMPL_H
#define	TNLNEIGHBOURGRIDENTITYGETTER2D_IMPL_H

#include <mesh/grids/tnlNeighbourGridEntityGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       2         |              2            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2 >, 2 >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int stepX, int stepY >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX,
                                                      entity.getCoordinates().y() + stepY ) );
      }
      
      template< int step >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return enityIndex + stepY * grid.getDimensions().x() + stepX;
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       2         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2 >, 1 >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      
      template< int stepX, int stepY >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( ! stepX + ! stepY == 1,
                    cerr << "Only one of the steps can be non-zero: stepX = " << stepX << " stepY = " << stepY );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), 
                                        stepY + ( stepY < 0 ) ) 
                       < grid.getDimensions() + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                      entity.getCoordinates().y() + stepY + ( stepY < 0 ) ),
                                     EntityOrientationType( stepX > 0 ? 1 : -1,
                                                            stepY > 0 ? 1 : -1 ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getNeighbourEntity< stepX, stepY >( grid, entity ) );
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       2         |            0              |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2 >, 0 >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;      
      
      template< int stepX, int stepY >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( stepX != 0 && stepY != 0,
                    cerr << " stepX = " << stepX << " stepY = " << stepY );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) 
                       < grid.getDimensions() + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " grid.getDimensions() + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ) ) = " 
                   << grid.getDimensions()  + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ) )
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                      entity.getCoordinates().y() + stepY + ( stepY < 0 ) ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getNeighbourEntity< stepX, stepY >( grid, entity ) );
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       1         |              2            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 1 >, 2 >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      
      template< int stepX, int stepY >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                    ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ),
                    cerr << "( stepX, stepY ) cannot be perpendicular to entity coordinates: stepX = " << stepX << " stepY = " << stepY
                         << " entity.getOrientation() = " << entity.getOrientation() ));
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + entity.getOrientation(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() + entity.getOrientation() = " << grid.getDimensions() + entity.getOrientation()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ), stepY - ( stepY > 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ), stepY - ( stepY > 0 ) ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX - ( stepX > 0 ),
                                                      entity.getCoordinates().y() + stepY - ( stepY > 0 ) ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getNeighbourEntity< stepX, stepY >( grid, entity ) );
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       0         |              0            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 0 >, 0 >
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int stepX, int stepY >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX,
                                                      entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return enityIndex + stepY * ( grid.getDimensions().x() + 1 ) + stepX;
      }
};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER2D_IMPL_H */

