/***************************************************************************
                          tnl-benchmark-simple-heat-equation-bug.h  -  description
                             -------------------
    begin                : Nov 28, 2015
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

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <core/tnlCuda.h>
#include <core/tnlHost.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlObject.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlLogger.h>
#include <fstream>
#include <iomanip>
#include <core/tnlAssert.h>
#include <mesh/tnlGnuplotWriter.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public tnlObject
{
};

template< typename Grid,          
          int EntityDimensions,
          typename Config >
class tnlGridEntity
{
};

template< typename Grid,
          typename GridEntity,
          int EntityDimensions = GridEntity::entityDimensions >
class tnlGridEntityGetter
{
   //static_assert( false, "Wrong mesh type or entity topology." );
};

//#include <mesh/grids/tnlGridEntityMeasureGetter.h>
template< typename Grid,
          int EntityDimensions >
class tnlGridEntityMeasureGetter
{   
};

enum tnlGridEntityStencilStorage
{ 
   tnlGridEntityNoStencil = 0,
   tnlGridEntityCrossStencil,
   tnlGridEntityFullStencil
};

template< int storage >
class tnlGridEntityStencilStorageTag
{
   public:
      
      static const int stencilStorage = storage;
};
/***
 * Common implementation for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlGridEntityMeasureGetter< tnlGrid< Dimensions, Real, Device, Index >, 0 >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
            
      template< typename EntityType >
      static const Real& getMeasure( const GridType& grid,
                                     const EntityType& entity )
      {
         return 0.0;
      }
};

/****
 * 2D grid
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityMeasureGetter< tnlGrid< 2, Real, Device, Index >, 2 >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      
      template< typename EntityType >
      static const Real& getMeasure( const GridType& grid,
                                     const EntityType& entity )
      {
         return grid.template getSpaceStepsProducts< 1, 1 >();
      }
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGridEntityMeasureGetter< tnlGrid< 2, Real, Device, Index >, 1 >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      
      template< typename EntityType >
      static const Real& getMeasure( const GridType& grid,
                                     const EntityType& entity )
      {
         if( entity.getOrientation().x() )
            return grid.template getSpaceStepsProducts< 0, 1 >();
         else
            return grid.template getSpaceStepsProducts< 1, 0 >();
      }
};


//#include <mesh/tnlDimensionsTag.h>
template< int Dimensions >
class tnlDimensionsTag
{
   public:

      static const int value = Dimensions;

      typedef tnlDimensionsTag< Dimensions - 1 > Decrement;

      static_assert( value >= 0, "The value of the dimensions cannot be negative." );
};

template<>
class tnlDimensionsTag< 0 >
{
   public:
   
      static const int value = 0;
};


//#include <mesh/grids/tnlNeighbourGridEntityGetter.h>

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename EntityStencilTag = 
            tnlGridEntityStencilStorageTag< GridEntity::ConfigType::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimensions ) > >
class tnlNeighbourGridEntityGetter
{
   public:

      // TODO: not all specializations are implemented yet
      
      __cuda_callable__
      tnlNeighbourGridEntityGetter( const GridEntity& entity )
      {
         //tnlAssert( false, );
      };
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex )
      {
         //tnlAssert( false, );
      };
};

//#include <mesh/grids/tnlNeighbourGridEntityGetter2D_impl.h>
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
      NeighbourGridEntityType getEntity() const
      {
         return NeighbourGridEntityType( this->grid,
                                         CoordinatesType( entity.getCoordinates().x() + stepX,
                                                          entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
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
      NeighbourGridEntityType getEntity() const
      {
            return NeighbourGridEntityType( this->entity.getMesh(),
                                            CoordinatesType( entity.getCoordinates().x() + stepX,
                                                             entity.getCoordinates().y() + stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA         
         if( ( stepX != 0 && stepY != 0 ) ||
             ( stepX < -stencilSize || stepX > stencilSize ||
               stepY < -stencilSize || stepY > stencilSize ) )         
            return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
         if( stepY == 0 )
            return stencilX[ stepX + stencilSize ];
         return stencilY[ stepY + stencilSize ];
#else
         return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
#endif         
         
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
                  entityIndex + index * neighbourEntityGetter.entity.getMesh().getDimensions().x();
            }
      };

      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA
         tnlStaticFor< IndexType, -stencilSize, 0, StencilYRefresher >::exec( *this, entityIndex );
         tnlStaticFor< IndexType, 1, stencilSize + 1, StencilYRefresher >::exec( *this, entityIndex );
         tnlStaticFor< IndexType, -stencilSize, stencilSize + 1, StencilXRefresher >::exec( *this, entityIndex );
#endif
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
      NeighbourGridEntityType getEntity() const
      {
         return NeighbourGridEntityType( this->entity.getMesh(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ),
                                         EntityOrientationType( stepX > 0 ? 1 : -1,
                                                                stepY > 0 ? 1 : -1 ),
                                         EntityBasisType( ! stepX, ! stepY ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetter::getEntityIndex( this->entity.getMesh(), this->template getEntity< stepX, stepY >() );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
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
      NeighbourGridEntityType getEntity() const
      {
         return NeighbourGridEntityType( this->grid,
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ) );
      }
      
      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetter::getEntityIndex( this->entity.getMesh(), this->template getEntity< stepX, stepY >() );
      }
      
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlNeighbourGridEntityGetter(){};      
};


//#include <mesh/grids/tnlNeighbourGridEntitiesStorage.h>
template< typename GridEntity,
          int NeighbourEntityDimensions >
class tnlNeighbourGridEntityLayer 
: public tnlNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 >
{   
   public:
      
      typedef tnlNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 > BaseType;
      typedef tnlNeighbourGridEntityGetter< GridEntity, NeighbourEntityDimensions > NeighbourEntityGetterType;
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      tnlNeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity ),
        BaseType( entity ) 
      {}
      
      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const tnlDimensionsTag< NeighbourEntityDimensions>& tag ) const
      {
         return this->neighbourEntities;
      }
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
         neighbourEntities.refresh( grid, entityIndex );
      };
      
   protected:
      
      NeighbourEntityGetterType neighbourEntities;
};

template< typename GridEntity >
class tnlNeighbourGridEntityLayer< GridEntity, 0 >
{
   public:
      
      typedef tnlNeighbourGridEntityGetter< GridEntity, 0 > NeighbourEntityGetterType;     
      
      __cuda_callable__
      tnlNeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity )
      {}

      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const tnlDimensionsTag< 0 >& tag ) const
      {
         return this->neighbourEntities;
      }
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         neighbourEntities.refresh( grid, entityIndex );
      };
      
   protected:
      
      NeighbourEntityGetterType neighbourEntities;
   
};

template< typename GridEntity >
class tnlNeighbourGridEntitiesStorage
: public tnlNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions >
{
   typedef tnlNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions > BaseType;
   
   public:
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      tnlNeighbourGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}

      
      template< int EntityDimensions >      
      __cuda_callable__
      const tnlNeighbourGridEntityGetter< GridEntity, EntityDimensions >&
      getNeighbourEntities() const
      {
         return BaseType::getNeighbourEntities( tnlDimensionsTag< EntityDimensions >() );
      }            

      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
      };
      
};


//#include <mesh/grids/tnlGridEntityConfig.h>

/****
 * This class says what neighbour grid entity indexes shall be pre-computed and stored in the
 * grid entity structure. If neighbourEntityStorage() returns false, nothing is stored.
 * Otherwise, if neighbour entity storage is enabled, we may store either only neighbour entities in a cross like this
 *
 *                X
 *   X            X
 *  XOX    or   XXOXX   etc.
 *   X            X
 *                X
 * 
 * or all neighbour entities like this
 * 
 *           XXXXX
 *  XXX      XXXXX
 *  XOX  or  XXOXX  etc.
 *  XXX      XXXXX
 *           XXXXX
 */

class tnlGridEntityNoStencilStorage
{
   public:
      
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( int neighbourEntityStorage )
      {
         return tnlGridEntityNoStencil;
      }
      
      constexpr static int getStencilSize()
      {
         return 0;
      }
};

template< int stencilSize = 1 >
class tnlGridEntityCrossStencilStorage
{
   public:
      
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( const int neighbourEntityDimensions )
      {
         //return tnlGridEntityNoStencil;
         return ( GridEntity::entityDimensions == GridEntity::GridType::meshDimensions &&
                  neighbourEntityDimensions == GridEntity::GridType::meshDimensions )
                * tnlGridEntityCrossStencil;         
      }
            
      constexpr static int getStencilSize()
      {
         return stencilSize;
      }
};



//#include <mesh/grids/tnlGridEntity.h>
template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter;

template< typename GridEntityType >
class tnlBoundaryGridEntityChecker;

template< typename GridEntityType >
class tnlGridEntityCenterGetter;



template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,          
          int EntityDimensions,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = EntityDimensions;
};

/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = 0;      
};


/****
 * Specializations for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = meshDimensions;

      constexpr static int getDimensions() { return entityDimensions; };
      
      constexpr static int getMeshDimensions() { return meshDimensions; };
      
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlGridEntity< GridType, entityDimensions, Config > ThisType;
      typedef tnlNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlNeighbourGridEntityGetter<
            tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;


      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid )
      : grid( grid ),
        entityIndex( -1 ),
        neighbourEntitiesStorage( *this )
      {
         this->coordinates = CoordinatesType( ( Index ) 0 );
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
            
      __cuda_callable__ inline
      tnlGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) )
      : grid( grid ),
        entityIndex( -1 ),
        coordinates( coordinates ),
        neighbourEntitiesStorage( *this )
      {
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
      
      
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const { return this->coordinates; };      
      
      __cuda_callable__ inline
      CoordinatesType& getCoordinates()  { return this->coordinates; };  
      
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates ) {this->coordinates = coordinates;};

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh(){
         this->entityIndex = this->grid.getEntityIndex( *this );
         this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
      };

      __cuda_callable__ inline
      Index getIndex() const { return this->entityIndex; };
            
      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const {return EntityOrientationType( ( IndexType ) 0 );};
      
      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation ){};
      
      __cuda_callable__ inline
      const EntityBasisType getBasis() const { return EntityBasisType( ( IndexType ) 1 );};
      
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};
      
      template< int NeighbourEntityDimensions = Dimensions >
      __cuda_callable__ inline
      const NeighbourEntities< NeighbourEntityDimensions >&
      getNeighbourEntities() const { return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimensions >();};
      
      __cuda_callable__ inline
      bool isBoundaryEntity() const { return tnlBoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );};
      
      __cuda_callable__ inline
      VertexType getCenter() const;
      
      __cuda_callable__ inline
      const RealType& getMeasure() const;      
      
      __cuda_callable__ inline
      const VertexType& getEntityProportions() const;      
      
      __cuda_callable__ inline
      const GridType& getMesh() const;

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
      //__cuda_callable__ inline
      //tnlGridEntity();
      
      friend class tnlBoundaryGridEntityChecker< ThisType >;
      
      friend class tnlGridEntityCenterGetter< ThisType >;
};


//#include <mesh/grids/tnlGridEntity.h>
/****
 * Specialization for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename tnlGrid< Dimensions, Real, Device, Index >::VertexType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getCenter() const
{
   return tnlGridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::RealType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getMeasure() const
{
   return this->getMesh().getCellMeasure();
}


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::VertexType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getEntityProportions() const
{
   return grid.getSpaceSteps();
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::GridType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getMesh() const
{
   return this->grid;
}


//#include <mesh/grids/tnlGridEntityGetter_impl.h>
/****
 * 2D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      
      static const int entityDimensions = 2;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         const CoordinatesType dimensions = grid.getDimensions();         

         return GridEntity
            ( grid,
              CoordinatesType( index % dimensions.x(),
                               index / dimensions.x() ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 1, 1 ) );
      }
      
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         //const CoordinatesType coordinates = entity.getCoordinates();
         //const CoordinatesType dimensions = grid.getDimensions();
         
         return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
      }
      
      
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
      
      static const int entityDimensions = 1;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions, EntityConfig > GridEntity;
      
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {         
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % aux, index / aux ),
                 typename GridEntity::EntityOrientationType( 1, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxFaces;
         const IndexType& aux = dimensions.x();
         return GridEntity
            ( grid,
              CoordinatesType( i % aux, i / aux ),
              typename GridEntity::EntityOrientationType( 0, 1 ),
              typename GridEntity::EntityBasisType( 1, 0 ) );
      }
      
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {                  
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         if( entity.getOrientation().x() )
            return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
         return grid.numberOfNxFaces + coordinates.y() * dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlGridEntityGetter< tnlGrid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
      
      static const int entityDimensions = 0;
      
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimensions > GridEntity;
      
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         const CoordinatesType dimensions = grid.getDimensions();

         const IndexType aux = dimensions.x() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( index % aux, 
                               index / aux ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0 ) );
      }
      
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {         
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
         
         return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
      }
};



template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 2, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 2, Real > VertexType;
   typedef tnlStaticVector< 2, Index > CoordinatesType;
   typedef tnlGrid< 2, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > CudaType;   
   typedef tnlGrid< 2, Real, Device, Index > ThisType;
   
   static const int meshDimensions = 2;

   template< int EntityDimensions, 
             typename Config = tnlGridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   using MeshEntity = tnlGridEntity< ThisType, EntityDimensions, Config >;
   
   typedef MeshEntity< meshDimensions, tnlGridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< meshDimensions - 1, tnlGridEntityNoStencilStorage > Face;
   typedef MeshEntity< 0 > Vertex;
      
   static constexpr int getMeshDimensions() { return meshDimensions; };

   tnlGrid();

   __cuda_callable__
   inline const CoordinatesType& getDimensions() const;

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );
   __cuda_callable__
   inline const VertexType& getOrigin() const;

   __cuda_callable__
   inline const VertexType& getProportions() const;

   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const;
   
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const;
   
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const;

   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
      
   __cuda_callable__
   RealType getCellMeasure() const;
   
   __cuda_callable__
   inline VertexType getSpaceSteps() const;

   template< int xPow, int yPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;
   
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;

   
   protected:

   __cuda_callable__
   void computeSpaceSteps();

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   VertexType origin, proportions;
   
   VertexType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ][ 5 ];
  
   template< typename, typename, int > 
   friend class tnlGridEntityGetter;
};


template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 2, Real, Device, Index > :: tnlGrid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlGrid< 2, Real, Device, Index > :: computeSpaceSteps()
{
   /*if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      const RealType& hx = this->spaceSteps.x(); 
      const RealType& hy = this->spaceSteps.y();
      
      Real auxX, auxY;
      for( int i = 0; i < 5; i++ )
      {
         switch( i )
         {
            case 0:
               auxX = 1.0 / ( hx * hx );
               break;
            case 1:
               auxX = 1.0 / hx;
               break;
            case 2:
               auxX = 1.0;
               break;
            case 3:
               auxX = hx;
               break;
            case 4:
               auxX = hx * hx;
               break;
         }
         for( int j = 0; j < 5; j++ )
         {
            switch( j )
            {
               case 0:
                  auxY = 1.0 / ( hy * hy );
                  break;
               case 1:
                  auxY = 1.0 / hy;
                  break;
               case 2:
                  auxY = 1.0;
                  break;
               case 3:
                  auxY = hy;
                  break;
               case 4:
                  auxY = hy * hy;
                  break;
            }
            this->spaceStepsProducts[ i ][ j ] = auxX * auxY;         
         }
      }
   }*/
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGrid< 2, Real, Device, Index >::CoordinatesType&
tnlGrid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDomain( const VertexType& origin,
                                                     const VertexType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename tnlGrid< 2, Real, Device, Index >::VertexType&
tnlGrid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename tnlGrid< 2, Real, Device, Index > :: VertexType&
   tnlGrid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
tnlGrid< 2, Real, Device, Index >:: 
getEntitiesCount() const
{
   static_assert( EntityType::entityDimensions <= 2 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
   
   switch( EntityType::entityDimensions )
   {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;         
      case 0:
         return this->numberOfVertices;
   }            
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
EntityType
tnlGrid< 2, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimensions <= 2 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
tnlGrid< 2, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimensions <= 2 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
tnlGrid< 2, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return tnlGridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real
tnlGrid< 2, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1 >();
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename tnlGrid< 2, Real, Device, Index >::VertexType
tnlGrid< 2, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow, int yPow  >
__cuda_callable__ inline 
const Real& 
tnlGrid< 2, Real, Device, Index >::
getSpaceStepsProducts() const
{
   tnlAssert( xPow >= -2 && xPow <= 2, 
              cerr << " xPow = " << xPow );
   tnlAssert( yPow >= -2 && yPow <= 2, 
              cerr << " yPow = " << yPow );

   return this->spaceStepsProducts[ yPow + 2 ][ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real tnlGrid< 2, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->spaceSteps.x(), this->spaceSteps.y() );
}





template< typename GridType, typename GridEntity >
__global__ void testKernel( const GridType* grid )
{   
   GridEntity entity( *grid );
}

int main( int argc, char* argv[] )
{
   const int gridXSize( 256 );
   const int gridYSize( 256 );        
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                      gridYSize / 16 + ( gridYSize % 16 != 0 ) );
      
   typedef tnlGrid< 2, double, tnlCuda > GridType;
   typedef typename GridType::VertexType VertexType;
   typedef typename GridType::CoordinatesType CoordinatesType;      
   GridType grid;
   GridType* cudaGrid;
   cudaMalloc( ( void** ) &cudaGrid, sizeof( GridType ) );
   cudaMemcpy( cudaGrid, &grid, sizeof( GridType ), cudaMemcpyHostToDevice );
   
   int iteration( 0 );
   auto t_start = std::chrono::high_resolution_clock::now();
   while( iteration < 1000 )
   {
      testKernel< GridType, typename GridType::Cell ><<< cudaGridSize, cudaBlockSize >>>( cudaGrid );
      cudaThreadSynchronize();
      iteration++;
   }
   auto t_stop = std::chrono::high_resolution_clock::now();   
   cudaFree( cudaGrid );
   
   std::cout << "Elapsed time = "
             << std::chrono::duration<double, std::milli>(t_stop-t_start).count() << std::endl;
   
   return EXIT_SUCCESS;   
}
