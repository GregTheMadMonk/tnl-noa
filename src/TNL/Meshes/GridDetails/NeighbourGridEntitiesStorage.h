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

namespace TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighbourEntityDimensions >
class NeighbourGridEntityLayer
: public NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 >
{
   public:
 
      typedef NeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 > BaseType;
      typedef NeighbourGridEntityGetter< GridEntity, NeighbourEntityDimensions > NeighbourEntityGetterType;
 
      using BaseType::getNeighbourEntities;
 
      __cuda_callable__
      NeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity ),
        BaseType( entity )
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
      };
 
   protected:
 
      NeighbourEntityGetterType neighbourEntities;
};

template< typename GridEntity >
class NeighbourGridEntityLayer< GridEntity, 0 >
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
      };
 
   protected:
 
      NeighbourEntityGetterType neighbourEntities;
 
};

template< typename GridEntity >
class NeighbourGridEntitiesStorage
: public NeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions >
{
   typedef NeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions > BaseType;
 
   public:
 
      using BaseType::getNeighbourEntities;
 
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

      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
      };      
};

} // namespace Meshes
} // namespace TNL

