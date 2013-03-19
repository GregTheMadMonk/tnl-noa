/***************************************************************************
                          tnlMersonSolverTester.h
                             -------------------
    begin                : Feb 1, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLMERSONSOLVERTESTER_H_
#define TNLMERSONSOLVERTESTER_H_

#include <legacy/mesh/tnlGridOld.h>
#include <solvers/ode/tnlMersonSolver.h>
#include <core/mfilename.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>

void heatEquationRHS( const int gridDimX,
                      const int gridDimY,
                      const int blockDimX,
                      const int blockDimY,
                      const int xSize,
                      const int ySize,
                      const float& hX,
                      const float& hY,
                      const float* u,
                      float* fu );

void heatEquationRHS( const int gridDimX,
                      const int gridDimY,
                      const int blockDimX,
                      const int blockDimY,
                      const int xSize,
                      const int ySize,
                      const double& hX,
                      const double& hY,
                      const double* u,
                      double* fu );

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void heatEquationRHSKernel( const Index xSize,
                                       const Index ySize,
                                       const Real hX,
                                       const Real hY,
                                       const Real* u,
                                       Real* fu )
{
   const Index i = blockIdx. x * blockDim. x + threadIdx. x;
   const Index j = blockIdx. y * blockDim. y + threadIdx. y;

   if( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 )
      fu[ xSize * j + i ] = 0.0;
   else
      if( i < xSize && j < ySize )
      fu[ xSize * j + i ] = ( u[ xSize * j + i + 1 ] - 2.0 * u[ xSize * j + i ] + u[ xSize * j + i - 1] ) / ( hX * hX ) +
                             ( u[ xSize * j + i + xSize ] - 2.0 * u[ xSize * j + i ] + u[ xSize * j + i - xSize ] ) / ( hY * hY );

}
#endif


template< typename Real, typename Device, typename Index = int >
class tnlMersonSolverTester : public CppUnit :: TestCase
{
   public:

   tnlMersonSolverTester( ){};

   tnlMersonSolverTester( const tnlString& s ){};

   tnlString getType() const
   {
      return tnlString( "tnlMersonSolverTester< " ) +
             GetParameterType( Real( 0 ) ) +
             tnlString( ", ") +
             Device :: getDeviceType() +
             tnlString( ", ") +
             GetParameterType( Index( 0 ) ) +
             tnlString( " >" );
   };

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMersonSolverTester" );
      CppUnit :: TestResult result;

      Real param;
      tnlString test_name = tnlString( "testUpdateU< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMersonSolverTester< Real, Device, Index > >(
               test_name. getString(),
               & tnlMersonSolverTester< Real, Device, Index > :: testUpdateU )
      );

      return suiteOfTests;
   };

   void GetExplicitRHS( const Real& time,
                        tnlGridOld< 2, Real, tnlHost, int >& u,
                        tnlGridOld< 2, Real, tnlHost, int >& fu )
   {
      const Index xSize = u. getDimensions(). x();
      const Index ySize = u. getDimensions(). y();
      for( Index i = 0; i < xSize; i ++ )
         for( Index j = 0; j < ySize; j ++ )
         {
            if( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 )
               fu( i, j ) = 0.0;
            else
               fu( i, j ) = u. Partial_xx( i, j ) + u. Partial_yy( i, j );
         }
   }

   void GetExplicitRHS( const Real& time,
                        tnlGridOld< 2, Real, tnlCuda, int >& u,
                        tnlGridOld< 2, Real, tnlCuda, int >& fu )
   {
#ifdef HAVE_CUDA
      const Index xSize = u. getDimensions(). x();
      const Index ySize = u. getDimensions(). y();
      const Index desBlockXSize = 16;
      const Index desBlockYSize = 16;
      const Index gridXSize = xSize / desBlockXSize + ( xSize % desBlockXSize != 0 );
      const Index gridYSize = ySize / desBlockYSize + ( ySize % desBlockYSize != 0 );
      const Real hX = u. getSpaceSteps(). x();
      const Real hY = u. getSpaceSteps(). y();
      dim3 gridDim( gridXSize, gridYSize );
      dim3 blockDim( desBlockXSize, desBlockYSize );
      heatEquationRHSKernel<<< gridDim, blockDim >>>( xSize, ySize, hX, hY, u. getData(), fu. getData() );
#endif
   }

   void testUpdateU()
   {
      const Index size = 128;

      tnlGridOld< 2, Real, tnlHost, int > hostU( "hostU");
      hostU. setDimensions( tnlTuple< 2, Index >( size, size ) );
      hostU. setDomain( tnlTuple< 2, Real >( 0.0, 0.0 ),
                        tnlTuple< 2, Real >( 1.0, 1.0 ) );
      const Real hx = hostU. getSpaceSteps(). x();
      const Real hy = hostU. getSpaceSteps(). y();
      for( Index i = 0; i < size; i ++ )
         for( Index j = 0; j < size; j ++ )
         {
            Real x = i * hx - 0.5;
            Real y = j * hy - 0.5;
            hostU( i, j ) = Sign( 0.25 - sqrt( x * x + y * y ) );
         }
      hostU. draw( "u-ini", "gnuplot" );

      tnlMersonSolver< tnlMersonSolverTester< Real, Device, Index > >
                       mersonSolver( "mersonSolver" );
      mersonSolver. setVerbosity( 2 );
      mersonSolver. setAdaptivity( 0.001 );
      mersonSolver. setTime( 0.0 );
      mersonSolver. setTau( 0.001 );


      tnlGridOld< 2, Real, tnlCuda, int > deviceU( "deviceU" );
      deviceU. setLike( hostU );
      deviceU = hostU;

      tnlGridOld< 2, Real, tnlHost, int > hostAuxU( "hostAuxU" );
      hostAuxU. setLike( hostU );

#ifdef HAVE_CUDA
      /*tnlMersonSolver< tnlMersonSolverTester< Real, Device, Index >,
                       tnlGridOld< 2, Real, tnlCuda, Index >,
                       Real,
                       tnlCuda,
                       Index > mersonSolverCUDA( "mersonSolverCuda" );
      mersonSolverCUDA. setVerbosity( 2 );
      mersonSolverCUDA. setAdaptivity( 0.001 );
      mersonSolverCUDA. setTime( 0.0 );
      mersonSolverCUDA. setTau( 0.001 );*/
#endif

      const Real finalTime = 0.1;
      const Real tau = 0.01;
      Real time = tau;
      int iteration( 0 );
      while( time < finalTime )
      {
         iteration ++;
         cout << "Starting the Merson solver with stop time " << time << endl;
         mersonSolver. setStopTime( time );
         mersonSolver. solve( *this, hostU );
         cout << "Starting the CUDA Merson solver with stop time " << time << endl;
#ifdef HAVE_CUDA
         //mersonSolverCUDA. setStopTime( time );
         //mersonSolverCUDA. solve( *this, deviceU );
#endif

         hostAuxU = deviceU;
         Real l1Norm = hostU. getDifferenceLpNorm( hostAuxU, ( Real ) 1.0 );
         Real l2Norm =  hostU. getDifferenceLpNorm( hostAuxU, ( Real ) 2.0 );
         Real maxNorm = hostU. getDifferenceAbsMax( hostAuxU );
         cout << endl;
         cout << "Errors: L1 " << l1Norm << " L2 " << l2Norm << " max." << maxNorm << endl;

         cout << "Writing file ... ";
         tnlString fileName;
         FileNameBaseNumberEnding(
                  "u",
                  iteration,
                  5,
                  ".gplt",
                  fileName );
         cout << fileName << endl;
         hostAuxU. draw( fileName, "gnuplot" );


         time += tau;
      }
   };
};

#endif /* TNLMERSONSOLVERCUDATESTER_H_ */
