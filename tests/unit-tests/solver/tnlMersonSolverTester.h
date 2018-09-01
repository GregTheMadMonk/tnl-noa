/***************************************************************************
                          MersonTester.h
                             -------------------
    begin                : Feb 1, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef MersonTESTER_H_
#define MersonTESTER_H_

#include <TNL/legacy/mesh/GridOld.h>
#include <TNL/Solvers/ODE/Merson.h>
#include <TNL/FileName.h>
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
class MersonTester : public CppUnit :: TestCase
{
   public:

   MersonTester( ){};

   MersonTester( const String& s ){};

   String getType() const
   {
      return String( "MersonTester< " ) +
             getType( Real( 0 ) ) +
             String( ", ") +
             Device :: getDeviceType() +
             String( ", ") +
             getType( Index( 0 ) ) +
             String( " >" );
   };

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "MersonTester" );
      CppUnit :: TestResult result;

      Real param;
      String test_name = String( "testUpdateU< " ) + getType( param ) + String( " >" );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< MersonTester< Real, Device, Index > >(
               test_name. getString(),
               & MersonTester< Real, Device, Index > :: testUpdateU )
      );

      return suiteOfTests;
   };

   void getExplicitUpdate( const Real& time,
                        GridOld< 2, Real, Devices::Host, int >& u,
                        GridOld< 2, Real, Devices::Host, int >& fu )
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

   void getExplicitUpdate( const Real& time,
                        GridOld< 2, Real, Devices::Cuda, int >& u,
                        GridOld< 2, Real, Devices::Cuda, int >& fu )
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

      GridOld< 2, Real, Devices::Host, int > hostU( "hostU");
      hostU. setDimensions( StaticVector< 2, Index >( size, size ) );
      hostU. setDomain( StaticVector< 2, Real >( 0.0, 0.0 ),
                        StaticVector< 2, Real >( 1.0, 1.0 ) );
      const Real hx = hostU. getSpaceSteps(). x();
      const Real hy = hostU. getSpaceSteps(). y();
      for( Index i = 0; i < size; i ++ )
         for( Index j = 0; j < size; j ++ )
         {
            Real x = i * hx - 0.5;
            Real y = j * hy - 0.5;
            hostU( i, j ) = sign( 0.25 - ::sqrt( x * x + y * y ) );
         }
      hostU. draw( "u-ini", "gnuplot" );

      Merson< MersonTester< Real, Device, Index > >
                       mersonSolver( "mersonSolver" );
      mersonSolver. setVerbosity( 2 );
      mersonSolver. setAdaptivity( 0.001 );
      mersonSolver. setTime( 0.0 );
      mersonSolver. setTau( 0.001 );


      GridOld< 2, Real, Devices::Cuda, int > deviceU( "deviceU" );
      deviceU. setLike( hostU );
      deviceU = hostU;

      GridOld< 2, Real, Devices::Host, int > hostAuxU( "hostAuxU" );
      hostAuxU. setLike( hostU );

#ifdef HAVE_CUDA
      /*Merson< MersonTester< Real, Device, Index >,
                       GridOld< 2, Real, Devices::Cuda, Index >,
                       Real,
                       Devices::Cuda,
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
        std::cout << "Starting the Merson solver with stop time " << time << std::endl;
         mersonSolver. setStopTime( time );
         mersonSolver. solve( *this, hostU );
        std::cout << "Starting the CUDA Merson solver with stop time " << time << std::endl;
#ifdef HAVE_CUDA
         //mersonSolverCUDA. setStopTime( time );
         //mersonSolverCUDA. solve( *this, deviceU );
#endif

         hostAuxU = deviceU;
         Real l1Norm = hostU. getDifferenceLpNorm( hostAuxU, ( Real ) 1.0 );
         Real l2Norm =  hostU. getDifferenceLpNorm( hostAuxU, ( Real ) 2.0 );
         Real maxNorm = hostU. getDifferenceAbsMax( hostAuxU );
        std::cout << std::endl;
        std::cout << "Errors: L1 " << l1Norm << " L2 " << l2Norm << " max." << maxNorm << std::endl;

        std::cout << "Writing file ... ";
         String fileName;
         FileNameBaseNumberEnding(
                  "u",
                  iteration,
                  5,
                  ".gplt",
                  fileName );
        std::cout << fileName << std::endl;
         hostAuxU. draw( fileName, "gnuplot" );


         time += tau;
      }
   };
};

#endif /* MersonCUDATESTER_H_ */
