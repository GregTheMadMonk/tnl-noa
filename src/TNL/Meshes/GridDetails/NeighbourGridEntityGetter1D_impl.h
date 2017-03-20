/***************************************************************************
                          NeighbourGridEntityGetter1D_impl.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/StaticFor.h>

namespace TNL {
namespace Meshes {

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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   1,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:
 
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;
 
      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
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
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   1,
   StencilStorage >
{
   public:
 
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;
      typedef NeighbourGridEntityGetter< GridEntityType, 1, StencilStorage > ThisType;
 
      static const int stencilSize = Config::getStencilSize();
 
      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( this->entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    this->entity.getCoordinates() < this->entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << this->entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
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
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
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
         StaticFor< IndexType, -stencilSize, stencilSize + 1, StencilRefresher >::exec( *this, entityIndex );
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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:
 
      static const int EntityDimensions = 1;
      static const int NeighbourEntityDimensions = 0;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;
 
      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 ).x() &&
                    entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions().x(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return this->entity.getIndex() + step + ( step < 0 );
      }
 
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
 
   protected:

      const GridEntityType& entity;
 
      //NeighbourGridEntityGetter(){};
 
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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   1,
   StencilStorage > //GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:
 
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;
 
      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
 
      void test() const { std::cerr << "***" << std::endl; };
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= 0 &&
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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   1,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:
 
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;
 
      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         return NeighbourGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }
 
      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
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
class NeighbourGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:
 
      static const int EntityDimensions = 0;
      static const int NeighbourEntityDimensions = 0;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimensions, Config > GridEntityType;
      typedef GridEntity< GridType, NeighbourEntityDimensions, Config > NeighbourGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighbourGridEntityType > GridEntityGetterType;

      __cuda_callable__ inline
      NeighbourGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
 
      template< int step >
      __cuda_callable__ inline
      NeighbourGridEntityType getEntity() const
      {
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
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
         TNL_ASSERT( entity.getCoordinates() >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimensions = " << EntityDimensions );
         TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
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

} // namespace Meshes
} // namespace TNL

