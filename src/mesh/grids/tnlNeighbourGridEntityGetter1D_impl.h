/***************************************************************************
                          tnlNeighbourGridEntityGetter1D_impl.h  -  description
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

#ifndef TNLNEIGHBOURGRIDENTITYGETTER1D_IMPL_H
#define	TNLNEIGHBOURGRIDENTITYGETTER1D_IMPL_H

#include <mesh/grids/tnlNeighbourGridEntityGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       1         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1 >, 1 >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int step >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );
         return enityIndex + step;
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       1         |              0            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1 >, 0 >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int step >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );

         return enityIndex + step + ( step < 0 );
      }
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimensions |       
 * +-----------------+---------------------------+
 * |       0         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0 >, 1 >
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int step >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );

         return enityIndex + step - ( step > 0 );
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
class tnlNeighbourGridEntityGetter< tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0 >, 0 >
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      template< int step >
      __cuda_callable__
      static NeighbourGridEntityType getNeighbourEntity( const GridType& grid,
                                                         const GridEntityType& entity )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__
      static IndexType getNeighbourEntityIndex( const GridType& grid,
                                                const GridEntityType& entity,
                                                const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << entityDimensions );
         tnlAssert( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) );

         return enityIndex + step;
      }
};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER1D_IMPL_H */

