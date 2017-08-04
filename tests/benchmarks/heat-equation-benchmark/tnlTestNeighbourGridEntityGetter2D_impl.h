/***************************************************************************
                          tnlTestNeighborGridEntityGetter2D_impl.h  -  description
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

#pragma once

#include "tnlTestNeighborGridEntityGetter.h"
#include <mesh/grids/Grid2D.h>
#include <core/tnlStaticFor.h>

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            | No specialization |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlTestNeighborGridEntityGetter< 
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   2,
   StencilStorage >
{
   public:
      
      static const int EntityDimension = 2;
      static const int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetter;

      __cuda_callable__ inline
      tnlTestNeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
            
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlTestNeighborGridEntityGetter(){};      
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlTestNeighborGridEntityGetter< 
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   2,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:
      
      static const int EntityDimension = 2;
      static const int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetter;
      typedef GridEntityStencilStorageTag< GridEntityCrossStencil > StencilStorage;
      typedef tnlTestNeighborGridEntityGetter< GridEntityType, 2, StencilStorage > ThisType;
      
      
      static const int stencilSize = Config::getStencilSize();

      __cuda_callable__ inline
      tnlTestNeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
      
      
      template< IndexType index > 
      class StencilXRefresher
      {
         public:
            
            __cuda_callable__
            static void exec( ThisType& neighborEntityGetter, const IndexType& entityIndex )
            {
               neighborEntityGetter.stencilX[ index + stencilSize ] = entityIndex + index;
            }
      };

      template< IndexType index > 
      class StencilYRefresher
      {
         public:
            
            __cuda_callable__
            static void exec( ThisType& neighborEntityGetter, const IndexType& entityIndex )
            {
               neighborEntityGetter.stencilY[ index + stencilSize ] = 
                  entityIndex + index * neighborEntityGetter.entity.getMesh().getDimensions().x();
            }
      };

      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA
         StaticFor< IndexType, -stencilSize, 0, StencilYRefresher >::exec( *this, entityIndex );
         StaticFor< IndexType, 1, stencilSize + 1, StencilYRefresher >::exec( *this, entityIndex );
         StaticFor< IndexType, -stencilSize, stencilSize + 1, StencilXRefresher >::exec( *this, entityIndex );
#endif
      };
      
   protected:

      const GridEntityType& entity;
      
      IndexType stencilX[ 2 * stencilSize + 1 ];
      IndexType stencilY[ 2 * stencilSize + 1 ];
      
      //tnlTestNeighborGridEntityGetter(){};      
};

