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
__device__ void computeBlockResidue( Real* du,
                                     Real* blockResidue,
                                     Index n )
{
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
    {
       if( blockDim.x >= 512 )
       {
          if( threadIdx.x < 256 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 256 ];
          __syncthreads();
       }
       if( blockDim.x >= 256 )
       {
          if( threadIdx.x < 128 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 128 ];
          __syncthreads();
       }
       if( blockDim.x >= 128 )
       {
          if( threadIdx.x < 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 64 ];
          __syncthreads();
       }

       /***
        * This runs in one warp so it is synchronized implicitly.
        */
       if ( threadIdx.x < 32)
       {
          if( blockDim.x >= 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 32 ];
          if( blockDim.x >= 32 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 16 ];
          if( blockDim.x >= 16 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 8 ];
          if( blockDim.x >=  8 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 4 ];
          if( blockDim.x >=  4 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 2 ];
          if( blockDim.x >=  2 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 1 ];
       }
    }
    else
    {
       int s;
       if( n >= 512 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 256 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 128 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 64 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();

       }
       if( n >= 32 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       /***
        * This runs in one warp so it is synchronised implicitly.
        */
       if( n >= 16 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 8 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 4 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 2 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
    }

   if( threadIdx.x == 0 )
      blockResidue[ blockIdx.x ] = du[ 0 ];

}

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
      aux[ c ] = tau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                       ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
   }
}

template< typename Real, typename Index >
__global__ void updateKernel( Real* u,
                              Real* aux,
                              Real* cudaBlockResidue,
                              const Index dofs )
{
   const Index blockOffset = blockIdx.x * blockDim.x;
   Index idx = blockOffset + threadIdx.x;
   
   if( idx < dofs )
      u[ idx ] += aux[ idx ];
   
   __syncthreads();

   const Index rest = dofs - blockOffset;
   Index n =  rest < blockDim.x ? rest : blockDim.x;

   computeBlockResidue< Real, Index >( aux,
                                       cudaBlockResidue,
                                       n );
}

template< typename Real, typename Index >
bool writeFunction(
   char* fileName,
   const Real* data,
   const Index xSize,
   const Index ySize,
   const Real& hx,
   const Real& hy )
{
   fstream file;
   file.open( fileName, ios::out );
   if( ! file )
   {
      cerr << "Unable to open file " << fileName << "." << endl;
      return false;
   }
   for( Index i = 0; i < xSize; i++ )
   {
      for( Index j = 0; j < ySize; j++ )
         file << i * hx << " " << j * hy << " " << data[ j * xSize + i ] << endl;
      file << endl;
   }
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
   const Index dofsCount = gridXSize * gridYSize;
   dim3 cudaUpdateBlocks( dofsCount / 256 + ( dofsCount % 256 != 0 ) );
   dim3 cudaUpdateBlockSize( 256 );
   
   /****
    * Initiation
    */   
   Real* u = new Real[ dofsCount ];
   Real* aux = new Real[ dofsCount ];
   Real* max_du = new Real[ cudaUpdateBlocks.x ];
   if( ! u || ! aux )
   {
      cerr << "I am not able to allocate grid function for grid size " << gridXSize << "x" << gridYSize << "." << endl;
      return false;
   }
   
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
   writeFunction( "initial", u, gridXSize, gridYSize, hx, hy );
   
   /****
    * Allocate data on the CUDA device
    */
   int cudaErr;
   Real *cuda_u, *cuda_aux, *cuda_max_du;
   cudaMalloc( &cuda_u, dofsCount * sizeof( Real ) );
   cudaMalloc( &cuda_aux, dofsCount * sizeof( Real ) );
   cudaMemcpy( cuda_u, u, dofsCount * sizeof( Real ),  cudaMemcpyHostToDevice );
   cudaMalloc( &cuda_max_du, cudaUpdateBlocks.x * sizeof( Real ) );
   if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
   {
      cerr << "Allocation failed. " << cudaErr << endl;
      return false;
   }
   
   /****
    * Explicit Euler solver
    */
   const int maxCudaGridSize = tnlCuda::getMaxGridSize();
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                      gridYSize / 16 + ( gridYSize % 16 != 0 ) );
   cout << "Setting grid size to " << cudaGridSize.x << "," << cudaGridSize.y << "," << cudaGridSize.z << endl;
   cout << "Setting block size to " << cudaBlockSize.x << "," << cudaBlockSize.y << "," << cudaBlockSize.z << endl;

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
      //cout << "Setting boundary conditions ... " << endl;
      boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( cuda_u, cuda_aux, gridXSize, gridYSize );
      if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
      {
         cerr << "Setting of boundary conditions failed. " << cudaErr << endl;
         return false;
      }
                    
      /****
       * Laplace operator
       */
      //cout << "Laplace operator ... " << endl;
      heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
         ( cuda_u, cuda_aux, tau, hx_inv, hy_inv, gridXSize, gridYSize );
      if( cudaGetLastError() != cudaSuccess )
      {
         cerr << "Laplace operator failed." << endl;
         return false;
      }
            
      /****
       * Update
       */            
      //cout << "Update ... " << endl;
      updateKernel<<< cudaUpdateBlocks, cudaUpdateBlockSize >>>( cuda_u, cuda_aux, cuda_max_du, dofsCount );
      if( cudaGetLastError() != cudaSuccess )
      {
         cerr << "Update failed." << endl;
         return false;
      }
      cudaThreadSynchronize();
      cudaMemcpy( max_du, cuda_max_du, cudaUpdateBlocks.x * sizeof( Real ), cudaMemcpyDeviceToHost );
      if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
      {
         cerr << "Copying max_du failed. " << cudaErr << endl;
         return false;
      }
      Real absMax( 0.0 );
      for( Index i = 0; i < cudaUpdateBlocks.x; i++ )
      {
         const Real a = fabs( max_du[ i ] );
         absMax = a > absMax ? a : absMax;
      }
            
      time += currentTau;
      iteration++;
      if( verbose && iteration % 1000 == 0 )
         cout << "Iteration: " << iteration << "\t Time:" << time << "    \r" << flush;
   }
   timer.stop();
   if( verbose )      
      cout << endl << "Finished..." << endl;
   cout << "Computation time is " << timer.getTime() << " sec. i.e. " << timer.getTime() / ( double ) iteration << "sec. per iteration." << endl;
   cudaMemcpy( u, cuda_u, dofsCount * sizeof( Real ), cudaMemcpyDeviceToHost );
   writeFunction( "final", u, gridXSize, gridYSize, hx, hy );
   
   /***
    * Freeing allocated memory
    */
   if( verbose )
      cout << "Freeing allocated memory..." << endl;
   delete[] u;
   delete[] aux;
   delete[] max_du;
   cudaFree( cuda_u );
   cudaFree( cuda_aux );
   cudaFree( cuda_max_du );
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
      if( verbose && iteration % 10000 == 0 )
         cout << "Iteration: " << iteration << "\t \t Time:" << time << "    \r" << flush;
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

