/***************************************************************************
                          tnlTestNeighbourGridEntitiesStorage.h  -  description
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
#include <mesh/MeshDimensionTag.h>
#include "tnlTestNeighbourGridEntityGetter.h"

template< typename GridEntity,
          int NeighbourEntityDimension >
class tnlTestNeighbourGridEntityLayer 
: public tnlTestNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimension - 1 >
{   
   public:
      
      typedef tnlTestNeighbourGridEntityLayer< GridEntity, NeighbourEntityDimension - 1 > BaseType;
      typedef tnlTestNeighbourGridEntityGetter< GridEntity, NeighbourEntityDimension > NeighbourEntityGetterType;
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      tnlTestNeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity ),
        BaseType( entity ) 
      {}
            
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
class tnlTestNeighbourGridEntityLayer< GridEntity, 0 >
{
   public:
      
      typedef tnlTestNeighbourGridEntityGetter< GridEntity, 0 > NeighbourEntityGetterType;     
      
      __cuda_callable__
      tnlTestNeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity )
      {}
      
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
class tnlTestNeighbourGridEntitiesStorage
: public tnlTestNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimension >
{
   typedef tnlTestNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimension > BaseType;
   
   public:
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      tnlTestNeighbourGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}

      

      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
      };
      
};


