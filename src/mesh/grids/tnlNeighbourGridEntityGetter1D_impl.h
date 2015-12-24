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
#include <core/tnlStaticFor.h>

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              1            |       ----        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1, Config >,
   1,
   tnlGridEntityStencilStorageTag< tnlGridEntityNoStencil > >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step;
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};      
      
   protected:

      const GridEntityType& entity;         
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              1            |  Cross/Full       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1, Config >,
   1,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      typedef tnlNeighbourGridEntityGetter< GridEntityType, 1, StencilStorage > ThisType;
      
      static const int stencilSize = Config::getStencilSize();
      
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         if( step < -stencilSize || step > stencilSize )
            return this->entity.getIndex() + step;
         return stencil[ stencilSize + step ];
      }
     
      template< IndexType index > 
      class StencilRefresher
      {
         public:
            
            __cuda_callable__
            static void exec( ThisType& neighbourEntityGetter, const IndexType& entityIndex )
            {
               neighbourEntityGetter.stencil[ index + stencilSize ] = entityIndex + index;
            }
      };
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
         tnlStaticFor< IndexType, -stencilSize, stencilSize + 1, StencilRefresher >::exec( *this, entityIndex );
      };      
      
   protected:

      const GridEntityType& entity;
      
      IndexType stencil[ 2 * stencilSize + 1 ];
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |       
 * +-----------------+---------------------------+-------------------+
 * |       1         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 1, Config >,
   0,
   tnlGridEntityStencilStorageTag< tnlGridEntityNoStencil > >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step + ( step < 0 );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};
      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |       
 * +-----------------+---------------------------+-------------------+
 * |       0         |              1            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0, Config >,
   1,
   tnlGridEntityStencilStorageTag< tnlGridEntityNoStencil > > 
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      
      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step - ( step > 0 );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};
      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |       
 * +-----------------+---------------------------+-------------------+
 * |       0         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0, Config >,
   0,
   tnlGridEntityStencilStorageTag< tnlGridEntityNoStencil > > 
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 1, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
      
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );

         return this->entity.getIndex() + step;
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};


#endif	/* TNLNEIGHBOURGRIDENTITYGETTER1D_IMPL_H */

