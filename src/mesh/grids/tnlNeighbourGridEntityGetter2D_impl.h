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
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            | No specialization |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2, Config >,
   2,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
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
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->grid,
                                         CoordinatesType( entity.getCoordinates().x() + stepX,
                                                          entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + stepY * entity.getGrid().getDimensions().x() + stepX;
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2, Config >,
   2,
   tnlGridEntityStencilStorageTag< tnlGridEntityCrossStencil > >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      typedef tnlGridEntityStencilStorageTag< tnlGridEntityCrossStencil > StencilStorage;
      typedef tnlNeighbourGridEntityGetter< GridEntityType, 2, StencilStorage > ThisType;
      
      
      static const int stencilSize = Config::getStencilSize();

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
            return NeighbourGridEntityType( this->grid,
                                            CoordinatesType( entity.getCoordinates().x() + stepX,
                                                             entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         if( ( stepX != 0 && stepY != 0 ) ||
             ( stepX < -stencilSize || stepX > stencilSize ||
               stepY < -stencilSize || stepY > stencilSize ) )         
            return this->entity.getIndex() + stepY * entity.getGrid().getDimensions().x() + stepX;
         if( stepY == 0 )
            return stencilX[ stepX + stencilSize ];
         return stencilY[ stepY + stencilSize ];
         
      }
      
      template< IndexType index > 
      class StencilXRefresher
      {
         public:
            
            __cuda_callable__
            static void exec( ThisType& neighbourEntityGetter, const IndexType& entityIndex )
            {
               neighbourEntityGetter.stencilX[ index + stencilSize ] = entityIndex + index;
            }
      };

      template< IndexType index > 
      class StencilYRefresher
      {
         public:
            
            __cuda_callable__
            static void exec( ThisType& neighbourEntityGetter, const IndexType& entityIndex )
            {
               neighbourEntityGetter.stencilY[ index + stencilSize ] = 
                  entityIndex + index * neighbourEntityGetter.entity.getGrid().getDimensions().x();
            }
      };

      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
         tnlStaticFor< IndexType, -stencilSize, 0, StencilYRefresher >::exec( *this, entityIndex );
         tnlStaticFor< IndexType, 1, stencilSize + 1, StencilYRefresher >::exec( *this, entityIndex );
         tnlStaticFor< IndexType, -stencilSize, stencilSize + 1, StencilXRefresher >::exec( *this, entityIndex );
      };
      
   protected:

      const GridEntityType& entity;
      
      IndexType stencilX[ 2 * stencilSize + 1 ];
      IndexType stencilY[ 2 * stencilSize + 1 ];
      
      //tnlNeighbourGridEntityGetter(){};      
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |       
 * +-----------------+---------------------------+-------------------+
 * |       2         |              1            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2, Config >,
   1,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 1;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      typedef typename GridEntityType::EntityBasisType EntityBasisType;

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( ! stepX + ! stepY == 1,
                    cerr << "Only one of the steps can be non-zero: stepX = " << stepX << " stepY = " << stepY );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), 
                                        stepY + ( stepY < 0 ) ) 
                       < entity.getGrid().getDimensions() + CoordinatesType( Sign( stepX ), Sign( stepY ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->entity.getGrid(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ),
                                         EntityOrientationType( stepX > 0 ? 1 : -1,
                                                                stepY > 0 ? 1 : -1 ),
                                         EntityBasisType( ! stepX, ! stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         return GridEntityGetter::getEntityIndex( this->entity.getGrid(), this->template getEntity< stepX, stepY >() );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |       
 * +-----------------+---------------------------+-------------------+
 * |       2         |            0              |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 2, Config >,
   0,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 2;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;      

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( stepX != 0 && stepY != 0,
                    cerr << " stepX = " << stepX << " stepY = " << stepY );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) 
                       < entity.getGrid().getDimensions() + CoordinatesType( Sign( stepX ), Sign( stepY ) ),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getGrid().getDimensions() + CoordinatesType( Sign( stepX ), Sign( stepY ) ) = " 
                   << entity.getGrid().getDimensions()  + CoordinatesType( Sign( stepX ), Sign( stepY ) )
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->grid,
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         return GridEntityGetter::getEntityIndex( this->entity.getGrid(), this->template getEntity< stepX, stepY >() );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |       
 * +-----------------+---------------------------+-------------------+
 * |       1         |              2            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 1, Config >,
   2,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 2;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlGridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef tnlGridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef tnlGridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetter;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                    ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ),
                    cerr << "( stepX, stepY ) cannot be perpendicular to entity coordinates: stepX = " << stepX << " stepY = " << stepY
                         << " entity.getOrientation() = " << entity.getOrientation() );
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() < entity.getGrid().getDimensions() + entity.getOrientation(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() + entity.getOrientation() = " << entity.getGrid().getDimensions() + entity.getOrientation()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ), stepY - ( stepY > 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + 
                       CoordinatesType( stepX - ( stepX > 0 ), stepY - ( stepY > 0 ) ) < entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->entity.getGrid(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX - ( stepX > 0 ),
                                                          entity.getCoordinates().y() + stepY - ( stepY > 0 ) ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         return GridEntityGetter::getEntityIndex( this->entity.getGrid(), this->template getEntity< stepX, stepY >() );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stencil Storage  |       
 * +-----------------+---------------------------+-------------------+
 * |       0         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< 2, Real, Device, Index >, 0, Config >,
   0,
   StencilStorage >
{
   public:
      
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef tnlGrid< 2, Real, Device, Index > GridType;
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
            
      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->grid,
                                         CoordinatesType( entity.getCoordinates().x() + stepX,
                                                          entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex()
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= entity.getGrid().getDimensions(),
              cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getGrid().getDimensions() = " << entity.getGrid().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + stepY * ( entity.getGrid().getDimensions().x() + 1 ) + stepX;
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};

#endif	/* TNLNEIGHBOURGRIDENTITYGETTER2D_IMPL_H */

