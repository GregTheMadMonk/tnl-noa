/***************************************************************************
                          tnlNeighbourGridEntityGetter3D_impl.h  -  description
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

#ifndef TNLNEIGHBOURGRIDENTITYGETTER3D_IMPL_H
#define	TNLNEIGHBOURGRIDENTITYGETTER3D_IMPL_H

#include <mesh/grids/tnlNeighbourGridEntityGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       3         |              3            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 3 >, 3 >
{
   public:
      
      static const int EntityDimensions = 3;
      static const int NeighbourEntityDimensions = 3;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX,
                                                      entity.getCoordinates().y() + stepY,
                                                      entity.getCoordinates().z() + stepZ ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = " 
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " 
                         << tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return enityIndex + ( stepZ * grid.getDimensions().y() + stepY ) * grid.getDimensions().x() + stepX;
      }

   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};      
      
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       3         |              2            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 3 >, 2 >
{
   public:
      
      static const int EntityDimensions = 3;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( ! stepX + ! stepY + ! stepZ == 2,
                    cerr << "Only one of the steps can be non-zero: stepX = " << stepX 
                         << " stepY = " << stepY
                         << " stepZ = " << stepZ );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), 
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) 
                       < grid.getDimensions() + 
                        CoordinatesType( tnlSign( stepX ), tnlSign( stepY ), tnlAbs(  stepZ ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                      entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                      entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ),
                                     EntityOrientationType( stepX > 0 ? 1 : -1,
                                                            stepY > 0 ? 1 : -1,
                                                            stepZ > 0 ? 1 : -1 ) );
      }
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getEntity< stepX, stepY, stepZ >( grid, entity ) );
      }
      
   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};            
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       3         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 3 >, 1 >
{
   public:
      
      static const int EntityDimensions = 3;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( ! stepX + ! stepY + ! stepZ == 1,
                    cerr << "Exactly two of the steps must be non-zero: stepX = " << stepX 
                         << " stepY = " << stepY
                         << " stepZ = " << stepZ );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), 
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) 
                       < grid.getDimensions() + 
                        CoordinatesType( tnlSign( stepX ), tnlSign( stepY ), tnlAbs(  stepZ ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                      entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                      entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ),
                                     EntityOrientationType( !!stepX, !!stepY, !!stepZ ),
                                     EntityBasisType( !stepX, !stepY, !stepZ ));
      }
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getEntity< stepX, stepY, stepZ >( grid, entity ) );
      }
      
   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};            
};


/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       3         |            0              |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 3 >, 0 >
{
   public:
      
      static const int EntityDimensions = 3;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;      

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( stepX != 0 && stepY != 0 && stepZ != 0,
                    cerr << " stepX = " << stepX 
                         << " stepY = " << stepY
                         << " stepZ = " << stepZ );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) 
                       < grid.getDimensions() + 
                            CoordinatesType( tnlSign( stepX ), tnlSign( stepY ), tnlSign( stepZ ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 )  ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " grid.getDimensions() + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ), tnlSign( stepZ ) ) = " 
                   << grid.getDimensions()  + CoordinatesType( tnlSign( stepX ), tnlSign( stepY ), tnlSign( stepZ ) ) )
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                      entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                      entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ) );
      }
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getEntity< stepX, stepY, stepZ >( grid, entity ) );
      }

   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};            
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       2         |              3            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 2 >, 3 >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 3;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                    ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ) &&
                    ( ( !! stepZ ) == ( !! entity.getOrientation().z() ) ),
                    cerr << "( stepX, stepY, stepZ ) cannot be perpendicular to entity coordinates: stepX = " << stepX
                         << " stepY = " << stepY << " stepZ = " << stepZ
                         << " entity.getOrientation() = " << entity.getOrientation() ));
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() < grid.getDimensions() + entity.getOrientation(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() + entity.getOrientation() = " << grid.getDimensions() + entity.getOrientation()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ),
                                        stepY - ( stepY > 0 ),
                                        stepZ - ( stepZ > 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ),
                                        stepY - ( stepY > 0 ),
                                        stepZ - ( stepZ > 0 ) ) < grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX - ( stepX > 0 ),
                                                      entity.getCoordinates().y() + stepY - ( stepY > 0 ),
                                                      entity.getCoordinates().z() + stepZ - ( stepZ > 0 ) ) );
      }
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return GridEntityGetter::getEntityIndex( grid,
            getEntity< stepX, stepY, stepZ >( grid, entity ) );
      }

   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};            
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
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 3, Real, Device, Index >, 0 >, 0 >
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) <= grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = " 
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + stepX,
                                                      entity.getCoordinates().y() + stepY,
                                                      entity.getCoordinates().z() + stepZ ) );
      }
      
      template< int stepZ, stepY, stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) <= grid.getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return entityIndex + stepZ * ( grid.getDimensions().y() + 1 + stepY ) * ( grid.getDimensions().x() + 1 ) + stepX;
      }

   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};      
      
};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER3D_IMPL_H */

