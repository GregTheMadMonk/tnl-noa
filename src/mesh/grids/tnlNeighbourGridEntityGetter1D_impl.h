/***************************************************************************
                          tnlNeighbourGridEntityGetter1D_impl.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/grids/tnlNeighbourGridEntityGetter.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>
#include <core/tnlStaticFor.h>

namespace TNL {

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
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
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
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntityType( this->entity.getMesh(), CoordinatesType( entity.getCoordinates().x() + step ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
#ifndef HAVE_CUDA  // TODO: fix it -- does not work with nvcc
         if( step < -stencilSize || step > stencilSize )
            return this->entity.getIndex() + step;
         return stencil[ stencilSize + step ];
#else
         return this->entity.getIndex() + step;
#endif
 
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
#ifndef HAVE_CUDA  // TODO: fix it -- does not work with nvcc
         tnlStaticFor< IndexType, -stencilSize, stencilSize + 1, StencilRefresher >::exec( *this, entityIndex );
#endif
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
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ).x() &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions().x(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
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
          typename Config,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter<
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0, Config >,
   1,
   StencilStorage > //tnlGridEntityStencilStorageTag< tnlGridEntityNoStencil > >
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
 
      void test() const { std::cerr << "***" << std::endl; };
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step - ( step > 0 ) >= 0 &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions().x(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step - ( step > 0 );
      }
 
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
};

/****   TODO: Implement this, now it is only a copy of specialization for none stencil storage
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighbourEntityDimensions |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       0         |              1            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlNeighbourGridEntityGetter<
   tnlGridEntity< tnlGrid< 1, Real, Device, Index >, 0, Config >,
   1,
   tnlGridEntityStencilStorageTag< tnlGridEntityCrossStencil > >
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
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step - ( step > 0 );
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
      NeighbourGridEntityType getEntity() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         tnlAssert( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );

         return this->entity.getIndex() + step;
      }
 
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

 
   protected:

      const GridEntityType& entity;
};

} // namespace TNL

