/***************************************************************************
                          NeighborGridEntitiesStorage.h  -  description
                             -------------------
    begin                : Dec 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>

namespace TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighborEntityDimension,
          typename GridEntityConfig,
          bool storage = GridEntityConfig::template neighborEntityStorage< GridEntity >( NeighborEntityDimension ) >
class NeighborGridEntityLayer{};   
   
template< typename GridEntity,
          int NeighborEntityDimension,
          typename GridEntityConfig >
class NeighborGridEntityLayer< GridEntity, NeighborEntityDimension, GridEntityConfig, true >
: public NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig >
{
   public:
 
      typedef NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig > BaseType;
      typedef NeighborGridEntityGetter< GridEntity, NeighborEntityDimension > NeighborEntityGetterType;

      using BaseType::getNeighborEntities;
 
      __cuda_callable__
      NeighborGridEntityLayer( const GridEntity& entity )
      : BaseType( entity ),
        neighborEntities( entity )
      {}
 
      __cuda_callable__
      const NeighborEntityGetterType& getNeighborEntities( const MeshDimensionTag< NeighborEntityDimension>& tag ) const
      {
         return this->neighborEntities;
      }
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
         neighborEntities.refresh( grid, entityIndex );
      }
 
   protected:
 
      NeighborEntityGetterType neighborEntities;
};

template< typename GridEntity,
          typename GridEntityConfig >
class NeighborGridEntityLayer< GridEntity, 0, GridEntityConfig, true >
{
   public:
 
      typedef NeighborGridEntityGetter< GridEntity, 0 > NeighborEntityGetterType;
 
      __cuda_callable__
      NeighborGridEntityLayer( const GridEntity& entity )
      : neighborEntities( entity )
      {}

      __cuda_callable__
      const NeighborEntityGetterType& getNeighborEntities( const MeshDimensionTag< 0 >& tag ) const
      {
         return this->neighborEntities;
      }
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         neighborEntities.refresh( grid, entityIndex );
      }
 
   protected:
 
      NeighborEntityGetterType neighborEntities;
};

template< typename GridEntity,
          int NeighborEntityDimension,
          typename GridEntityConfig >
class NeighborGridEntityLayer< GridEntity, NeighborEntityDimension, GridEntityConfig, false >
: public NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig >
{
   public:
      
      typedef NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig > BaseType;      
      typedef NeighborGridEntityGetter< GridEntity, NeighborEntityDimension > NeighborEntityGetterType;

      using BaseType::getNeighborEntities;
 
      __cuda_callable__
      NeighborGridEntityLayer( const GridEntity& entity )
      : BaseType( entity )
      {}

      __cuda_callable__
      const NeighborEntityGetterType& getNeighborEntities( const MeshDimensionTag< NeighborEntityDimension >& tag ) const {}
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex ) {}
};

template< typename GridEntity,
          typename GridEntityConfig >
class NeighborGridEntityLayer< GridEntity, 0, GridEntityConfig, false >
{
   public:
      
      typedef NeighborGridEntityGetter< GridEntity, 0 > NeighborEntityGetterType;
         
      __cuda_callable__
      NeighborGridEntityLayer( const GridEntity& entity ){}

      __cuda_callable__
      const NeighborEntityGetterType& getNeighborEntities( const MeshDimensionTag< 0 >& tag ) const {}
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex ) {}
};




template< typename GridEntity,
          typename GridEntityConfig >
class NeighborGridEntitiesStorage
: public NeighborGridEntityLayer< GridEntity, GridEntity::meshDimension, GridEntityConfig >
{
   typedef NeighborGridEntityLayer< GridEntity, GridEntity::meshDimension, GridEntityConfig > BaseType;
 
   public:
 
      using BaseType::getNeighborEntities;
      using BaseType::refresh;
 
      __cuda_callable__
      NeighborGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}
 
      template< int EntityDimension >
      __cuda_callable__
      const NeighborGridEntityGetter< GridEntity, EntityDimension >&
      getNeighborEntities() const
      {
         return BaseType::getNeighborEntities( MeshDimensionTag< EntityDimension >() );
      }
};


} // namespace Meshes
} // namespace TNL

