/***************************************************************************
                          TestNeighbourGridEntitiesStorage.h  -  description
                             -------------------
    begin                : Dec 18, 2015
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

#include <core/tnlCuda.h>
#include <mesh/tnlDimensionsTag.h>
#include "TestNeighbourGridEntityGetter.h"

template< typename GridEntity,
          int NeighbourEntityDimensions >
class TestNeighbourGridEntityLayer 
: public TestNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 >
{   
   public:
      
      typedef TestNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimensions - 1 > BaseType;
      typedef TestNeighbourGridEntityGetter< GridEntity, NeighbourEntityDimensions > NeighbourEntityGetterType;
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      TestNeighbourGridEntityLayer( const GridEntity& entity )
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
class TestNeighbourGridEntityLayer< GridEntity, 0 >
{
   public:
      
      typedef TestNeighbourGridEntityGetter< GridEntity, 0 > NeighbourEntityGetterType;     
      
      __cuda_callable__
      TestNeighbourGridEntityLayer( const GridEntity& entity )
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
class TestNeighbourGridEntitiesStorage
: public TestNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions >
{
   typedef TestNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions > BaseType;
   
   public:
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      TestNeighbourGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}

      
      template< int EntityDimensions >      
      __cuda_callable__
      const TestNeighbourGridEntityGetter< GridEntity, EntityDimensions >&
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



