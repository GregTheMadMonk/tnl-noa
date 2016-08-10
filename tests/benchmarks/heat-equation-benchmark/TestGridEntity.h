/***************************************************************************
                          GridEntity.h  -  description
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
 
template< typename GridEntity >
class TestNeighbourGridEntitiesStorage
{  
   public:
      
      __cuda_callable__
      TestNeighbourGridEntitiesStorage( const GridEntity& entity )
      : entity( entity )
      {}
      
      const GridEntity& entity;
};

template< typename Grid,          
          int EntityDimensions >
class TestGridEntity
{
};


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,          
          int EntityDimensions >
class TestGridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions >
{
   public:
      static const int entityDimensions = EntityDimensions;
};

/****
 * Specializations for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class TestGridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions >
{
   public:
      
      typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      //typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = meshDimensions;

      constexpr static int getDimensions() { return entityDimensions; };
      
      constexpr static int getMeshDimensions() { return meshDimensions; };
      
      
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef Vectors::StaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef TestGridEntity< GridType, entityDimensions > ThisType;
      typedef TestNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      __cuda_callable__ inline
      TestGridEntity( const GridType& grid )
      : grid( grid ),
        /*entityIndex( -1 ),*/
        neighbourEntitiesStorage( *this )
      {
      }
      
      
      __cuda_callable__ inline
      TestGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) )
      : grid( grid ),
        /*entityIndex( -1 ),
        coordinates( coordinates ),*/
        neighbourEntitiesStorage( *this )
        {
        }

      

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
};





