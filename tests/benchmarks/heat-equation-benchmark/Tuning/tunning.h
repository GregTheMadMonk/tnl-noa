/***************************************************************************
                          tunning.h  -  description
                             -------------------
    begin                : Aug 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
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

#ifdef HAVE_CUDA
#include<cuda.h>
#endif

/****
 * Just testing data for measuring performance
 * with different ways of passing data to kernels.
 */
struct Data
{
   double time, tau;
   TNL::Containers::StaticVector< 2, double > c1, c2, c3, c4;
   TNL::Meshes::Grid< 2, double > grid;
};

#ifdef HAVE_CUDA

template< typename UserData, typename Grid, typename Real, typename Index >
__global__ void _boundaryConditionsKernel( const Grid* grid,
                                           UserData userData )
{
   Real* u = userData.u;
   const typename UserData::BoundaryConditionsType* bc = userData.boundaryConditions;
   using Coordinates = typename Grid::CoordinatesType;
   const Index& gridXSize = grid->getDimensions().x();
   const Index& gridYSize = grid->getDimensions().y();
   Coordinates coordinates( ( blockIdx.x ) * blockDim.x + threadIdx.x,
                             ( blockIdx.y ) * blockDim.y + threadIdx.y );
   
   Index c = coordinates.y() * gridXSize + coordinates.x();
   
   if( coordinates.x() == 0 && coordinates.y() < gridYSize )
   {
      //aux[ coordinates.y() * gridXSize ] = 0.0;
      u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 ); //0.0;
   }
   if( coordinates.x() == gridXSize - 1 && coordinates.y() < gridYSize )
   {
      //aux[ coordinates.y() * gridXSize + gridYSize - 1 ] = 0.0;
      //userData.u[ coordinates.y() * gridXSize + gridXSize - 1 ] = 0.0; //userData.u[ coordinates.y() * gridXSize + gridXSize - 1 ];
      u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 ); //0.0;
   }
   if( coordinates.y() == 0 && coordinates.x() < gridXSize )
   {
      //aux[ coordinates.x() ] = 0.0; //userData.u[ coordinates.y() * gridXSize + 1 ];
      //userData.u[ coordinates.x() ] = 0.0; //userData.u[ coordinates.y() * gridXSize + 1 ];
      u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 ); //0.0;
   }
   if( coordinates.y() == gridYSize -1  && coordinates.x() < gridXSize )
   {
      //aux[ coordinates.y() * gridXSize + coordinates.x() ] = 0.0; //userData.u[ coordinates.y() * gridXSize + gridXSize - 1 ];      
      //userData.u[ coordinates.y() * gridXSize + coordinates.x() ] = 0.0; //userData.u[ coordinates.y() * gridXSize + gridXSize - 1 ];      
      u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 ); //0.0;
   }         
}

template< typename UserData, typename Grid, typename Real, typename Index >
__global__ void _heatEquationKernel( const Grid* grid,
                                     UserData userData )
{
   Real* u = userData.u;
   Real* fu = userData.fu;
   const typename UserData::DifferentialOperatorType* op = userData.differentialOperator;

   const Index& gridXSize = grid->getDimensions().x();
   const Index& gridYSize = grid->getDimensions().y();
   const Real& hx_inv = grid->template getSpaceStepsProducts< -2,  0 >();
   const Real& hy_inv = grid->template getSpaceStepsProducts<  0, -2 >();
   
   typename Grid::CoordinatesType 
      coordinates( blockIdx.x * blockDim.x + threadIdx.x, 
                   blockIdx.y * blockDim.y + threadIdx.y );
   if( coordinates.x() > 0 && coordinates.x() < gridXSize - 1 &&
       coordinates.y() > 0 && coordinates.y() < gridYSize - 1 )
   {
      const Index c = coordinates.y() * gridXSize + coordinates.x();
      fu[ c ] = ( *op )( *grid, u, c, coordinates, 0 );
         /*( ( userData.u[ c - 1 ]         - 2.0 * userData.u[ c ] + userData.u[ c + 1 ]         ) * hx_inv +
                   ( userData.u[ c - gridXSize ] - 2.0 * userData.u[ c ] + userData.u[ c + gridXSize ] ) * hy_inv );*/
   }  
}


#endif

