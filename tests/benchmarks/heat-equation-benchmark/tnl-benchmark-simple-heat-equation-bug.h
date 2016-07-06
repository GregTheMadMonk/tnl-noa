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
//#include <mesh/grids/tnlGridEntityConfig.h>
class tnlGridEntityNoStencilStorage
{
   public:
      
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( int neighbourEntityStorage )
      {
         return tnlGridEntityNoStencil;
      }
};

//#include <mesh/grids/tnlGridEntity.h>
template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename StencilStorage >
class tnlNeighbourGridEntityGetter;



//#include <mesh/grids/tnlNeighbourGridEntityGetter.h>

template< typename GridEntity,
          int NeighbourEntityDimensions,
          typename EntityStencilTag = 
            tnlGridEntityStencilStorageTag< GridEntity::ConfigType::template neighbourEntityStorage< GridEntity >( NeighbourEntityDimensions ) > >
class tnlNeighbourGridEntityGetter
{
   public:
      __cuda_callable__
      tnlNeighbourGridEntityGetter( const GridEntity& entity ){};
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex ){};
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

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity ){}
      //: entity( &entity )      {}
      
            
   protected:

      //const GridEntityType* entity;
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

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity ){}
      //: entity( &entity )     {}
      
   protected:

      //const GridEntityType* entity;
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

      __cuda_callable__ inline
      tnlNeighbourGridEntityGetter( const GridEntityType& entity ){}
      //: entity( &entity ){}
      
   protected:

     //const GridEntityType* entity;
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
            
      __cuda_callable__
      tnlNeighbourGridEntityLayer( const GridEntity& entity )
      : neighbourEntities( entity ),
        BaseType( entity ) 
      {}
      
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

   protected:
      
      NeighbourEntityGetterType neighbourEntities;
   
};

template< typename GridEntity >
class tnlNeighbourGridEntitiesStorage
: public tnlNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions >
{
   typedef tnlNeighbourGridEntityLayer< GridEntity, GridEntity::meshDimensions > BaseType;
   
   public:
      
      __cuda_callable__
      tnlNeighbourGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}
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
      
      /*template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlNeighbourGridEntityGetter<
            tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;*/


      __cuda_callable__ inline
      tnlGridEntity()
      : e1( this ), e2( this ), e3( this ),
        entityIndex( -1 ), eidx2( -1 ), eidx3( -1 )        
      {
         //this->coordinates = CoordinatesType( ( Index ) 0 );
         //this->orientation = EntityOrientationType( ( Index ) 0 );
         //this->basis = EntityBasisType( ( Index ) 1 );
      }
                  
   protected:
      
      //const GridType& grid;
      
      IndexType entityIndex, eidx2, eidx3;
      
      ThisType *e1, *e2, *e3;
      
      /*CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;*/
      
      //NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
      
      
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
   
   typedef MeshEntity< meshDimensions, tnlGridEntityNoStencilStorage > Cell; //tnlGridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< meshDimensions - 1, tnlGridEntityNoStencilStorage > Face;
   typedef MeshEntity< 0 > Vertex;
      
   static constexpr int getMeshDimensions() { return meshDimensions; };

   tnlGrid()
   : numberOfCells( 0 ), numberOfNxFaces( 0 ), numberOfNyFaces( 0 ), numberOfFaces( 0 ), numberOfVertices( 0 ) {}

     
   protected:

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   VertexType origin, proportions;
   
   VertexType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ][ 5 ];
  
   template< typename, typename, int > 
   friend class tnlGridEntityGetter;
};


template< typename GridType, typename GridEntity >
__global__ void testKernel( const GridType* grid )
{   
   GridEntity entity;
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
