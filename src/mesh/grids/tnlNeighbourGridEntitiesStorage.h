/***************************************************************************
                          tnlNeighbourGridEntitiesStorage.h  -  description
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


#ifndef TNLNEIGHBOURGRIDENTITIESSTORAGE_H
#define	TNLNEIGHBOURGRIDENTITIESSTORAGE_H

#include <core/tnlCuda.h>
#include <mesh/tnlDimensionsTag.h>

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
: public tnlNeighbourGridEntityLayer< GridEntity, GridEntity::entityDimensions >
{
   typedef tnlNeighbourGridEntityLayer< GridEntity, GridEntity::entityDimensions > BaseType;
   
   public:
      
      using BaseType::getNeighbourEntities;
      
      __cuda_callable__
      tnlNeighbourGridEntitiesStorage( const GridEntity& entity )
      : tnlNeighbourGridEntityLayer< GridEntity, GridEntity::entityDimensions >( entity )
      {}

      
      template< int EntityDimensions >      
      __cuda_callable__
      const tnlNeighbourGridEntityGetter< GridEntity, EntityDimensions >&
      getNeighbourEntities() const
      {
         return getNeighbourEntities( tnlDimensionsTag< EntityDimensions >() );
      }            

      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
      };
      
};

#endif	/* TNLNEIGHBOURGRIDENTIESSTORAGE_H */

