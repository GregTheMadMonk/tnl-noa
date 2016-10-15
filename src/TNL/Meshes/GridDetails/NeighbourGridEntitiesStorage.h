/***************************************************************************
                          NeighbourGridEntitiesStorage.h  -  description
                             -------------------
    begin                : Dec 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/Meshes/MeshDimensionsTag.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>

namespace TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename GridEntityConfig,
          bool storage = GridEntityConfig::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimensions ) >
class NeighbourGridEntityLayer{};   
   
template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename GridEntityConfig >
class NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions, GridEntityConfig, true >
: public NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1, GridEntityConfig >
{
   public:
 
      typedef NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1, GridEntityConfig > BaseType;
      typedef NeighbourGridEntityGetter< GridEntity, NeighbourEntityDimensions > NeighbourEntityGetterType;

      using BaseType::getNeighbourEntities;
 
      __cuda_callable__
      NeighbourGridEntityLayer( const GridEntity& entity )
      : BaseType( entity ),
        neighbourEntities( entity )
      {}
 
      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const MeshDimensionsTag< NeighbourEntityDimensions>& tag ) const
      {
         return this->neighbourEntities;
      }
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
         neighbourEntities.refresh( grid, entityIndex );
      }
 
   protected:
 
      NeighbourEntityGetterType neighbourEntities;
};

template< typename GridEntity,
          typename GridEntityConfig >
class NeighbourGridEntityLayer< GridEntity, 0, GridEntityConfig, true >
{
   public:
 
      typedef NeighbourGridEntityGetter< GridEntity, 0 > NeighbourEntityGetterType;
 
      __cuda_callable__
      NeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity )
      {}

      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const MeshDimensionsTag< 0 >& tag ) const
      {
         return this->neighbourEntities;
      }
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         neighbourEntities.refresh( grid, entityIndex );
      }
 
   protected:
 
      NeighbourEntityGetterType neighbourEntities;
};

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename GridEntityConfig >
class NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions, GridEntityConfig, false >
: public NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1, GridEntityConfig >
{
   public:
      
      typedef NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1, GridEntityConfig > BaseType;      
      typedef NeighbourGridEntityGetter< GridEntity, NeighbourEntityDimensions > NeighbourEntityGetterType;

      using BaseType::getNeighbourEntities;
 
      __cuda_callable__
      NeighbourGridEntityLayer( const GridEntity& entity )
      : BaseType( entity )
      {}

      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const MeshDimensionsTag< NeighbourEntityDimensions >& tag ) const {}
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex ) {}
};

template< typename GridEntity,
          typename GridEntityConfig >
class NeighbourGridEntityLayer< GridEntity, 0, GridEntityConfig, false >
{
   public:
      
      typedef NeighbourGridEntityGetter< GridEntity, 0 > NeighbourEntityGetterType;
         
      __cuda_callable__
      NeighbourGridEntityLayer( const GridEntity& entity ){}

      __cuda_callable__
      const NeighbourEntityGetterType& getNeighbourEntities( const MeshDimensionsTag< 0 >& tag ) const {}
 
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex ) {}
};




template< typename GridEntity,
          typename GridEntityConfig >
class NeighbourGridEntitiesStorage
: public NeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions, GridEntityConfig >
{
   typedef NeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions, GridEntityConfig > BaseType;
 
   public:
 
      using BaseType::getNeighbourEntities;
      using BaseType::refresh;
 
      __cuda_callable__
      NeighbourGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}
 
      template< int EntityDimensions >
      __cuda_callable__
      const NeighbourGridEntityGetter< GridEntity, EntityDimensions >&
      getNeighbourEntities() const
      {
         return BaseType::getNeighbourEntities( MeshDimensionsTag< EntityDimensions >() );
      }
};


} // namespace Meshes
} // namespace TNL

