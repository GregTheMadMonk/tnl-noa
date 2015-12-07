/***************************************************************************
                          tnl-benchmark-simple-heat-equation.h  -  description
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

#ifndef TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H
#define	TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H

#include <iostream>
#include <stdio.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlTimerRT.h>
#include <core/tnlCuda.h>

using namespace std;

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void boundaryConditionsKernel( const Real* u, Real* aux,
                                          const Index gridXSize, const Index gridYSize )
{
   const Index i = ( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index j = ( blockIdx.y ) * blockDim.y + threadIdx.y;
   if( i == 0 && j < gridYSize )
      aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   if( i == gridXSize - 1 && j < gridYSize )
      aux[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   if( j == 0 && i < gridXSize )
      aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   if( j == gridYSize -1  && i < gridXSize )
      aux[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
}


template< typename Real, typename Index >
__global__ void heatEquationKernel( const Real* u, 
                                    Real* aux,
                                    const Real tau,
                                    const Real hx_inv,
                                    const Real hy_inv,
                                    const Index gridXSize,
                                    const Index gridYSize )
{
   const Index i = blockIdx.x * blockDim.x + threadIdx.x;
   const Index j = blockIdx.y * blockDim.y + threadIdx.y;
   if( i > 0 && i < gridXSize - 1 &&
       j > 0 && j < gridYSize - 1 )
   {
      const Index c = j * gridXSize + i;
      aux[ c ] = u[ c ] + tau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                                  ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
   }
}

template< typename Real, typename Index >
__global__ void updateKernel( Real* u,
                              const Real* aux,
                              const Index dofs )
{
   Index idx = blockIdx.x * blockDim.x + threadIdx.x;
   if( idx < dofs )
      u[ idx ] += aux[ idx ];
}

template< typename Real, typename Index >
bool solveHeatEquationCuda( const tnlParameterContainer& parameters )
{
   const Real domainXSize = parameters.getParameter< double >( "domain-x-size" );
   const Real domainYSize = parameters.getParameter< double >( "domain-y-size" );
   const Index gridXSize = parameters.getParameter< int >( "grid-x-size" );
   const Index gridYSize = parameters.getParameter< int >( "grid-y-size" );
   const Real sigma = parameters.getParameter< double >( "sigma" );
   Real tau = parameters.getParameter< double >( "time-step" );
   const Real finalTime = parameters.getParameter< double >( "final-time" );
   const bool verbose = parameters.getParameter< bool >( "verbose" );
   
   /****
    * Initiation
    */   
   Real* u = new Real[ gridXSize * gridYSize ];
   Real* aux = new Real[ gridXSize * gridYSize ];
   if( ! u || ! aux )
   {
      cerr << "I am not able to allocate grid function for grid size " << gridXSize << "x" << gridYSize << "." << endl;
      return false;
   }
   const Index dofsCount = gridXSize * gridYSize;
   const Real hx = domainXSize / ( Real ) gridXSize;
   const Real hy = domainYSize / ( Real ) gridYSize;
   const Real hx_inv = 1.0 / ( hx * hx );
   const Real hy_inv = 1.0 / ( hy * hy );
   if( ! tau )
   {
      tau = hx * hx < hy * hy ? hx * hx : hy * hy;
      if( verbose )
         cout << "Setting tau to " << tau << "." << endl;
   }
   
   /****
    * Initial condition
    */
   if( verbose )
      cout << "Setting the initial condition ... " << endl;
   for( Index j = 0; j < gridYSize; j++ )
      for( Index i = 0; i < gridXSize; i++ )
      {
         const Real x = i * hx - domainXSize / 2.0;      
         const Real y = j * hy - domainYSize / 2.0;      
         u[ j * gridXSize + i ] = exp( - sigma * ( x * x + y * y ) );
      }
   
   /****
    * Allocate data on the CUDA device
    */
   
   Real *cuda_u, *cuda_aux;
   cudaMalloc( &cuda_u, gridXSize * gridYSize * sizeof( Real ) );
   cudaMalloc( &cuda_aux, gridXSize * gridYSize * sizeof( Real ) );
   cudaMemcpy( cuda_u, u, gridXSize * gridYSize * sizeof( Real ),  cudaMemcpyHostToDevice );
   
   /****
    * Explicit Euler solver
    */
   const int maxCudaGridSize = tnlCuda::getMaxGridSize();
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                      gridYSize / 16 + ( gridYSize % 16 != 0 ) );
   const Index dofs = gridXSize * gridYSize;

   if( verbose )
      cout << "Starting the solver main loop..." << endl;   
   tnlTimerRT timer;
   timer.reset();
   timer.start();
   Real time( 0.0 );   
   Index iteration( 0 );
   while( time < finalTime )
   {
      const Real timeLeft = finalTime - time;
      const Real currentTau = tau < timeLeft ? tau : timeLeft;      

      /****
       * Neumann boundary conditions
       */
      boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( u, aux, gridXSize, gridYSize );

      /****
       * Laplace operator
       */
      heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
         ( u, aux, tau, hx_inv, hy_inv, gridXSize, gridYSize );

      /****
       * Update
       */      
      updateKernel<<< dofs / 256 + ( dofs % 256 != 0 ), 256 >>>( u, aux, dofs );
      
      /*Real absMax( 0.0 );
      for( Index i = 0; i < dofsCount; i++ )
      {
         const Real a = fabs( aux[ i ] );
         absMax = a > absMax ? a : absMax;
      }*/
      cudaThreadSynchronize();
            
      time += currentTau;
      iteration++;
      if( verbose )
         cout << "Iteration: " << iteration << "\t Time:" << time << "    \r" << flush;
   }
   timer.stop();
   if( verbose )      
      cout << endl << "Finished..." << endl;
   cout << "Computation time is " << timer.getTime() << " sec. i.e. " << timer.getTime() / ( double ) iteration << "sec. per iteration." << endl;
   
   /***
    * Freeing allocated memory
    */
   if( verbose )
      cout << "Freeing allocated memory..." << endl;
   delete[] u;
   delete[] aux;
   cudaFree( cuda_u );
   cudaFree( cuda_aux );
   return true;
}
#endif

template< typename Real, typename Index >
bool solveHeatEquationHost( const tnlParameterContainer& parameters )
{
   const Real domainXSize = parameters.getParameter< double >( "domain-x-size" );
   const Real domainYSize = parameters.getParameter< double >( "domain-y-size" );
   const Index gridXSize = parameters.getParameter< int >( "grid-x-size" );
   const Index gridYSize = parameters.getParameter< int >( "grid-y-size" );
   const Real sigma = parameters.getParameter< double >( "sigma" );
   Real tau = parameters.getParameter< double >( "time-step" );
   const Real finalTime = parameters.getParameter< double >( "final-time" );
   const bool verbose = parameters.getParameter< bool >( "verbose" );
   
   /****
    * Initiation
    */   
   Real* u = new Real[ gridXSize * gridYSize ];
   Real* aux = new Real[ gridXSize * gridYSize ];
   if( ! u || ! aux )
   {
      cerr << "I am not able to allocate grid function for grid size " << gridXSize << "x" << gridYSize << "." << endl;
      return false;
   }
   const Index dofsCount = gridXSize * gridYSize;
   const Real hx = domainXSize / ( Real ) gridXSize;
   const Real hy = domainYSize / ( Real ) gridYSize;
   const Real hx_inv = 1.0 / ( hx * hx );
   const Real hy_inv = 1.0 / ( hy * hy );
   if( ! tau )
   {
      tau = hx * hx < hy * hy ? hx * hx : hy * hy;
      if( verbose )
         cout << "Setting tau to " << tau << "." << endl;
   }
   
   /****
    * Initial condition
    */
   if( verbose )
      cout << "Setting the initial condition ... " << endl;
   for( Index j = 0; j < gridYSize; j++ )
      for( Index i = 0; i < gridXSize; i++ )
      {
         const Real x = i * hx - domainXSize / 2.0;      
         const Real y = j * hy - domainYSize / 2.0;      
         u[ j * gridXSize + i ] = exp( - sigma * ( x * x + y * y ) );
      }
   
   /****
    * Explicit Euler solver
    */
   if( verbose )
      cout << "Starting the solver main loop..." << endl;
   tnlTimerRT timer;
   timer.reset();
   timer.start();
   Real time( 0.0 );   
   Index iteration( 0 );
   while( time < finalTime )
   {
      const Real timeLeft = finalTime - time;
      const Real currentTau = tau < timeLeft ? tau : timeLeft;

      /****
       * Neumann boundary conditions
       */
      for( Index j = 0; j < gridYSize; j++ )
      {
         aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
         aux[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
      }
      for( Index i = 0; i < gridXSize; i++ )
      {
         aux[ i ] = 0.0; //u[ gridXSize + i ];
         aux[ ( gridYSize - 1 ) * gridXSize + i ] = 0.0; //u[ ( gridYSize - 2 ) * gridXSize + i ];
      }
      
      /*for( Index j = 1; j < gridYSize - 1; j++ )
         for( Index i = 1; i < gridXSize - 1; i++ )
         {
            const Index c = j * gridXSize + i;
            aux[ c ] = u[ c ] + currentTau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                                               ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
         }
      Real* swap = aux;
      aux = u;
      u = swap;
      */

      for( Index j = 1; j < gridYSize - 1; j++ )
         for( Index i = 1; i < gridXSize - 1; i++ )
         {
            const Index c = j * gridXSize + i;
            aux[ c ] = currentTau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                                     ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
         }
      
      
      Real absMax( 0.0 );
      for( Index i = 0; i < dofsCount; i++ )
      {
         const Real a = fabs( aux[ i ] );
         absMax = a > absMax ? a : absMax;
      }
      
      for( Index i = 0; i < dofsCount; i++ )
         u[ i ] += aux[ i ];         
      
      time += currentTau;
      iteration++;
      if( verbose )
         cout << "Iteration: " << iteration << "\t Time:" << time << "    \r" << flush;
   }
   timer.stop();
   if( verbose )      
      cout << endl << "Finished..." << endl;
   cout << "Computation time is " << timer.getTime() << " sec. i.e. " << timer.getTime() / ( double ) iteration << "sec. per iteration." << endl;
   
   /***
    * Freeing allocated memory
    */
   if( verbose )
      cout << "Freeing allocated memory..." << endl;
   delete[] u;
   delete[] aux;
   return true;
}

int main( int argc, char* argv[] )
{
   tnlConfigDescription config;
   config.addEntry< tnlString >( "device", "Device the computation will run on.", "host" );
      config.addEntryEnum< tnlString >( "host" );
#ifdef HAVE_CUDA      
      config.addEntryEnum< tnlString >( "cuda" );
#endif      
   config.addEntry< int >( "grid-x-size", "Grid size along x-axis.", 100 );
   config.addEntry< int >( "grid-y-size", "Grid size along y-axis.", 100 );
   config.addEntry< double >( "domain-x-size", "Domain size along x-axis.", 2.0 );
   config.addEntry< double >( "domain-y-size", "Domain size along y-axis.", 2.0 );
   config.addEntry< double >( "sigma", "Sigma in exponential initial condition.", 2.0 );
   config.addEntry< double >( "time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
   config.addEntry< double >( "final-time", "Final time of the simulation.", 1.0 );
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
   
   tnlParameterContainer parameters;
   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;
   
   tnlString device = parameters.getParameter< tnlString >( "device" );
   if( device == "host" &&
       ! solveHeatEquationHost< double, int >( parameters  ) )
      return EXIT_FAILURE;
   if( device == "cuda" &&
       ! solveHeatEquationCuda< double, int >( parameters  ) )
   return EXIT_SUCCESS;   
}


#endif	/* TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H */

