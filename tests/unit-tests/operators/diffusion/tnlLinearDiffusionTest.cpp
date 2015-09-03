/***************************************************************************
                          tnlLinearDiffusionTest.cpp  -  description
                             -------------------
    begin                : Sep 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "../tnlPDEOperatorEocTester.h"
#include "../../tnlUnitTestStarter.h"
#include <mesh/tnlGrid.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>
#include "../tnlPDEOperatorEocTestResult.h"
#include <functors/tnlExpBumpFunction.h>

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename TestFunction,
          typename ApproximationMethod >
class tnlPDEOperatorEocTestResult< tnlLinearDiffusion< tnlGrid< Dimensions, Real, Device, Index >, Real, Index >,
                                   ApproximationMethod,
                                   TestFunction >
{
   public:
      static Real getL1Eoc() { return ( Real ) 2.0; };
      static Real getL1Tolerance() { return ( Real ) 0.05; };

      static Real getL2Eoc() { return ( Real ) 2.0; };
      static Real getL2Tolerance() { return ( Real ) 0.05; };

      static Real getMaxEoc() { return ( Real ) 2.0; };
      static Real getMaxTolerance() { return ( Real ) 0.05; };

};

int main( int argc, char* argv[] )
{
   const bool verbose( false );
   const int MeshSize( 32 );
#ifdef HAVE_CPPUNIT
   /****
    * Explicit approximation
    */
   if( ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 1, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 1 >,
                                                             tnlExpBumpFunction< 1, double >,
                                                             tnlExplicitApproximation,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 2, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 2 >,
                                                             tnlExpBumpFunction< 2, double >,
                                                             tnlExplicitApproximation,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 3, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 3 >,
                                                             tnlExpBumpFunction< 3, double >,
                                                             tnlExplicitApproximation,
                                                             MeshSize,
                                                             verbose > >()
                                                              )
      return EXIT_FAILURE;
   /****
    * Implicit (matrix) approximation
    */
   if( ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 1, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 1 >,
                                                             tnlExpBumpFunction< 1, double >,
                                                             tnlImplicitApproximation,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 2, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 2 >,
                                                             tnlExpBumpFunction< 2, double >,
                                                             tnlImplicitApproximation,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlLinearDiffusion< tnlGrid< 3, double, tnlHost, int >, double, int >,
                                                             tnlExactLinearDiffusion< 3 >,
                                                             tnlExpBumpFunction< 3, double >,
                                                             tnlImplicitApproximation,
                                                             MeshSize,
                                                             verbose > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}

