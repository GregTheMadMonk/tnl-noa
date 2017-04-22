/***************************************************************************
                          tnlTestGridEntity.h  -  description
                             -------------------
    begin                : Nov 13, 2015
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

template< typename Grid,          
          int EntityDimension,
          typename Config >
class tnlTestGridEntity
{
};

/****
 * Specializations for cells
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlTestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >
{
   public:
      
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimension = GridType::meshDimension;
      
      static const int entityDimension = meshDimension;

      constexpr static int getDimension() { return entityDimension; };
      
      constexpr static int getDimension() { return meshDimension; };
      
      
      typedef tnlStaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimension, IndexType > EntityBasisType;
      typedef tnlTestGridEntity< GridType, entityDimension, Config > ThisType;
      //typedef tnlTestNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      /*template< int NeighbourEntityDimension = entityDimension >
      using NeighbourEntities = 
         tnlTestNeighbourGridEntityGetter<
            tnlTestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           entityDimension,
                           Config >,
            NeighbourEntityDimension >;*/


      __cuda_callable__ inline
      tnlTestGridEntity( const GridType& grid )
      : grid( grid ),
        entityIndex( -1 )/*,
        neighbourEntitiesStorage( *this )*/
      {
         this->coordinates = CoordinatesType( ( Index ) 0 );
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
      
      
      __cuda_callable__ inline
      tnlTestGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) )
      : grid( grid ),
        entityIndex( -1 ),
        coordinates( coordinates )/*,
        neighbourEntitiesStorage( *this )*/
      {
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
      
      

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      //NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
};

