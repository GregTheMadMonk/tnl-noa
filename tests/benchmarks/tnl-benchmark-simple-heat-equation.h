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
#include <config/tnlParameterContainer.h>
#include <core/tnlTimerRT.h>

using namespace std;

template< typename Real, typename Index >
bool solveHeatEquation( const tnlParameterContainer& parameters )
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

#endif	/* TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H */

