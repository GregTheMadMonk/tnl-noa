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
#include <core/vectors/tnlStaticVector.h>
#include <mesh/tnlGrid.h>


using namespace std;

/*enum tnlGridEntityStencilStorage
{ 
   tnlGridEntityNoStencil = 0,
   tnlGridEntityCrossStencil,
   tnlGridEntityFullStencil
};*/

template< typename Grid,          
          int EntityDimensions,
          typename Config >
class tnlTestGridEntity
{
   protected:
      tnlTestGridEntity( const tnlTestGridEntity& e ){};
};


template< typename GridType >
class TestGridEntity
{
   public:      
      
      __device__ TestGridEntity( const GridType& grid )
      : grid( grid ), entity( *this )
      {}
      
   protected:
      
      const GridType& grid;
      
      const TestGridEntity& entity;
            
};

/*
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


template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid //: public tnlObject
{
};
 */

/****
 * Specializations for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlTestGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >
{
   public:
      
      typedef tnlGrid< Dimensions, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::VertexType VertexType;
      typedef Config ConfigType;
      
      static const int meshDimensions = 2; //GridType::meshDimensions;
      
      static const int entityDimensions = meshDimensions;

      constexpr static int getDimensions() { return entityDimensions; };
      
      constexpr static int getMeshDimensions() { return meshDimensions; };
      
      
      typedef tnlStaticVector< meshDimensions, IndexType > EntityOrientationType;
      typedef tnlStaticVector< meshDimensions, IndexType > EntityBasisType;
      typedef tnlTestGridEntity< GridType, entityDimensions, Config > ThisType;
      //typedef tnlTestNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
      
      /*template< int NeighbourEntityDimensions = entityDimensions >
      using NeighbourEntities = 
         tnlTestNeighbourGridEntityGetter<
            tnlTestGridEntity< tnlGrid< Dimensions, Real, Device, Index >,
                           entityDimensions,
                           Config >,
            NeighbourEntityDimensions >;*/


      __cuda_callable__ inline
      tnlTestGridEntity( const GridType& grid )
      : grid( grid ),
        entityIndex( -1 )/*,
        neighbourEntitiesStorage( *this )*/
      {
         this->coordinates = CoordinatesType( ( IndexType ) 0 );
         this->orientation = EntityOrientationType( ( IndexType ) 0 );
         this->basis = EntityBasisType( ( IndexType ) 1 );
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
         this->orientation = EntityOrientationType( ( IndexType ) 0 );
         this->basis = EntityBasisType( ( IndexType ) 1 );
      }
      
      

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      //NeighbourGridEntitiesStorageType neighbourEntitiesStorage;
      
};

#ifdef UNDEF
template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 2, Real, Device, Index > //: public tnlObject
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

   //template< int EntityDimensions, 
   //          typename Config = tnlGridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   //using MeshEntity = tnlGridEntity< ThisType, EntityDimensions, Config >;
   //
   //typedef MeshEntity< meshDimensions, tnlGridEntityCrossStencilStorage< 1 > > Cell;
   //typedef MeshEntity< meshDimensions - 1, tnlGridEntityNoStencilStorage > Face;
   //typedef MeshEntity< 0 > Vertex;
   

   // TODO: remove this
   template< int EntityDimensions, 
             typename Config = tnlGridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   using TestMeshEntity = tnlTestGridEntity< ThisType, EntityDimensions, Config >;
   typedef TestMeshEntity< meshDimensions, tnlGridEntityNoStencilStorage > Cell;
   //typedef TestMeshEntity< meshDimensions, tnlGridEntityCrossStencilStorage< 1 > > TestCell;
   /////
   
   static constexpr int getMeshDimensions() { return meshDimensions; };

   tnlGrid()
   : numberOfCells( 0 ),
     numberOfNxFaces( 0 ),
     numberOfNyFaces( 0 ),
     numberOfFaces( 0 ),
     numberOfVertices( 0 ) {};

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const {};

   void setDimensions( const Index xSize, const Index ySize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   inline const CoordinatesType& getDimensions() const { return this->dimensions; };

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );
   __cuda_callable__
   inline const VertexType& getOrigin() const {return this->origin;};

   __cuda_callable__
   inline const VertexType& getProportions() const { return this->proportions; };

   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const
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
   };
   
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const
   {
      static_assert( EntityType::entityDimensions <= 2 &&
                     EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );

      //return tnlGridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
   };
   
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const
   {
      static_assert( EntityType::entityDimensions <= 2 &&
                     EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );

      //return tnlGridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
   };

   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
      
   __cuda_callable__
   RealType getCellMeasure() const
   {
      return this->template getSpaceStepsProducts< 1, 1 >();
   };
   
   __cuda_callable__
   inline VertexType getSpaceSteps() const
   {
      return this->spaceSteps;
   };

   template< int xPow, int yPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const
   {
      tnlAssert( xPow >= -2 && xPow <= 2, 
                 cerr << " xPow = " << xPow );
      tnlAssert( yPow >= -2 && yPow <= 2, 
                 cerr << " yPow = " << yPow );

      return this->spaceStepsProducts[ yPow + 2 ][ xPow + 2 ];
   };
   
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;

   
   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool writeMesh( const tnlString& fileName,
                   const tnlString& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;


   protected:

   __cuda_callable__
   void computeSpaceSteps()
   {
      if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
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
      }
   };
   

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   VertexType origin, proportions;
   
   VertexType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ][ 5 ];
  
   template< typename, typename, int > 
   friend class tnlGridEntityGetter;
};

#endif

class tnlTestGrid
{
   public:
   
      typedef TestGridEntity< tnlTestGrid > Cell;
      //typedef tnlTestGridEntity< tnlTestGrid, 2, tnlGridEntityNoStencilStorage > Cell;
      typedef tnlStaticVector< 2, double > VertexType;
      typedef tnlStaticVector< 2, int > CoordinatesType;   

      CoordinatesType dimensions;

      int numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

      VertexType origin, proportions;

      VertexType spaceSteps;

      double spaceStepsProducts[ 5 ][ 5 ];
   
};







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
      
   //typedef tnlTestGrid GridType;
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
