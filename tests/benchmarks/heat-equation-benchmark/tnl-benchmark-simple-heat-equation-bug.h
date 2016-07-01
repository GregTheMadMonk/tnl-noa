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


class tnlTestGrid
{
   public:
   
      typedef TestGridEntity< tnlTestGrid > Cell;
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
