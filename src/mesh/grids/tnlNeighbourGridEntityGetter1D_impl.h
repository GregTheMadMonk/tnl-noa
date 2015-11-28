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
      typedef tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( ( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              ( tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ) );
         return entityIndex + step;
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
      typedef tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( ( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              ( tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ) );

         return entityIndex + step + ( step < 0 );
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
      typedef tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;
      
      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}

      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( ( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              ( tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ) );

         return entityIndex + step - ( step > 0 );
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
      typedef tnlGridEntityGetter< GridType, NeighbourEntityDimensions > GridEntityGetter;

      tnlNeighbourGridEntityGetter( const GridType& grid,
                                    const GridEntityType& entity )
      : grid( grid ),
        entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex( const IndexType entityIndex )
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= grid.getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " grid.getDimensions() = " << grid.getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( ( entityIndex == 
                    tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ),
                    cerr << "entityIndex = " << entityIndex 
                         << " tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) = " << 
                              ( tnlGridEntityGetter< GridType, EntityDimensions >::getEntityIndex( grid, entity ) ) );

         return entityIndex + step;
      }
   
   protected:

      const GridType& grid;

      const GridEntityType& entity;
      
      tnlNeighbourGridEntityGetter(){};      
};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER1D_IMPL_H */

