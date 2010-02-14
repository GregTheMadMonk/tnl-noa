/***************************************************************************
                          tnlMersonSolverCUDATester.h
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

#ifndef TNLMERSONSOLVERCUDATESTER_H_
#define TNLMERSONSOLVERCUDATESTER_H_

#include <diff/tnlMersonSolver.h>
#include <diff/tnlMersonSolverCUDA.h>
#include <diff/drawGrid2D.h>
#include <diff/norms.h>
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


template< class T > class tnlMersonSolverCUDATester : public CppUnit :: TestCase
{
   public:

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMersonSolverCUDATester" );
      CppUnit :: TestResult result;

      T param;
      tnlString test_name = tnlString( "testUpdateU< " ) + GetParameterType( param ) + tnlString( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMersonSolverCUDATester< T > >(
               test_name. Data(),
               & tnlMersonSolverCUDATester< T > :: testUpdateU )
      );

      return suiteOfTests;
   };

   void GetExplicitRHS( const T& time,
            tnlGrid2D< T >& u,
            tnlGrid2D< T >& fu )
   {
      const int xSize = u. GetXSize();
      const int ySize = u. GetYSize();
      for( int i = 0; i < xSize; i ++ )
         for( int j = 0; j < ySize; j ++ )
         {
            if( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 )
               fu( i, j ) = 0.0;
            else
               fu( i, j ) = u. Partial_xx( i, j ) + u. Partial_yy( i, j );
         }
   }

   void GetExplicitRHSCUDA( const T& time,
            tnlGridCUDA2D< T >& u,
            tnlGridCUDA2D< T >& fu )
   {
      const int xSize = u. GetXSize();
      const int ySize = u. GetYSize();
      const int desBlockXSize = 16;
      const int desBlockYSize = 16;
      const int gridXSize = xSize / desBlockXSize + ( xSize % desBlockXSize != 0 );
      const int gridYSize = ySize / desBlockYSize + ( ySize % desBlockYSize != 0 );
      heatEquationRHS( gridXSize,
               gridYSize,
               desBlockXSize,
               desBlockYSize,
               xSize,
               ySize,
               u. GetHx(),
               u. GetHy(),
               u. Data(),
               fu. Data() );
      /*tnlGrid2D< T > hostU;
      hostU. SetNewDimensions( u. GetXSize(), u. GetYSize() );
      hostU. SetNewDomain( u. GetAx(), u. GetBx(), u. GetAy(), u. GetBy() );
      tnlGrid2D< T > hostFu( hostU );
      hostU. copyFrom( u );
      hostFu. copyFrom( fu );
      Draw( hostU, "device-u", "gnuplot" );
      Draw( hostFu, "device-fu", "gnuplot" );

      GetExplicitRHS( time, hostU, hostFu );
      u. copyFrom( hostU );
      fu. copyFrom( hostFu );
      Draw( hostU, "host-u", "gnuplot" );
      Draw( hostFu, "host-fu", "gnuplot" );
      getchar();*/
   }


   void testUpdateU()
   {
      const int size = 1024;

      tnlGrid2D< T > hostU;
      hostU. SetNewDimensions( size, size );
      hostU. SetNewDomain( 0.0, 1.0, 0.0, 1.0 );
      const T hx = hostU. GetHx();
      const T hy = hostU. GetHy();
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
         {
            T x = i * hx - 0.5;
            T y = j * hy - 0.5;
            hostU( i, j ) = Sign( 0.25 - sqrt( x * x + y * y ) );
         }
      Draw( hostU, "u-ini", "gnuplot" );

      tnlGridCUDA2D< T > deviceU( hostU );
      deviceU. copyFrom( hostU );

      tnlGrid2D< T > hostAuxU( hostU );

      tnlMersonSolver< tnlGrid2D< T >, tnlMersonSolverCUDATester, T > mersonSolver( hostU );
      mersonSolver. SetVerbosity( 2 );
      mersonSolver. SetTime( 0.0 );
      mersonSolver. SetTau( 1.0 );

      tnlMersonSolverCUDA< tnlGridCUDA2D< T >, tnlMersonSolverCUDATester, T > mersonSolverCUDA( deviceU );
      mersonSolverCUDA. SetVerbosity( 2 );
      mersonSolverCUDA. SetTime( 0.0 );
      mersonSolverCUDA. SetTau( 1.0 );

      const T finalTime = 0.1;
      const T tau = 0.01;
      T time = tau;
      int iteration( 0 );
      while( time < finalTime )
      {
         iteration ++;
         //cout << "Starting the Merson solver with stop time " << time << endl;
         //mersonSolver. Solve( *this, hostU, time, 0.0 );
         //cout << "Starting the CUDA Merson solver with stop time " << time << endl;
         mersonSolverCUDA. Solve( *this, deviceU, time, 0.0 );

         hostAuxU. copyFrom( deviceU );
         /*T l1Norm = GetDiffL1Norm( hostU, hostAuxU );
         T l2Norm = GetDiffL2Norm( hostU, hostAuxU );
         T maxNorm = GetDiffMaxNorm( hostU, hostAuxU );
         cout << endl;
         cout << "Errors: L1 " << l1Norm << " L2 " << l2Norm << " max." << maxNorm << endl;*/

         cout << "Writing file ... ";
         tnlString fileName;
         FileNameBaseNumberEnding(
                  "u",
                  iteration,
                  5,
                  ".gplt",
                  fileName );
         cout << fileName << endl;
         Draw( hostAuxU, fileName. Data(), "gnuplot" );


         time += tau;
      }




   };
};

#endif /* TNLMERSONSOLVERCUDATESTER_H_ */
