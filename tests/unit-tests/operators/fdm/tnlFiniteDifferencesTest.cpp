/***************************************************************************
                          tnlFiniteDifferencesTest.cpp  -  description
                             -------------------
    begin                : Jan 12, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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
#include <operators/fdm/tnlBackwardFiniteDifference.h>
#include <operators/fdm/tnlForwardFiniteDifference.h>
#include <operators/fdm/tnlCentralFiniteDifference.h>
#include "../tnlPDEOperatorEocTestResult.h"
#include <functions/tnlExpBumpFunction.h>

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename TestFunction >
class tnlPDEOperatorEocTestResult< tnlFiniteDifferences< tnlGrid< Dimensions, Real, Device, Index >, Real, Index >,
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
   const bool verbose( true );
   const int MeshSize( 16 );
#ifdef HAVE_CPPUNIT
   /****
    * Explicit approximation
    */
   if( ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 1, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 1 >,
                                                             tnlExpBumpFunction< 1, double >,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 2, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 2 >,
                                                             tnlExpBumpFunction< 2, double >,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 3, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 3 >,
                                                             tnlExpBumpFunction< 3, double >,
                                                             MeshSize,
                                                             verbose > >()
                                                              )
      return EXIT_FAILURE;

#ifdef UNDEF   
   /****
    * Implicit (matrix) approximation
    */
   if( ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 1, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 1 >,
                                                             tnlExpBumpFunction< 1, double >,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 2, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 2 >,
                                                             tnlExpBumpFunction< 2, double >,
                                                             MeshSize,
                                                             verbose > >() ||
       ! tnlUnitTestStarter :: run< tnlPDEOperatorEocTester< tnlFiniteDifferences< tnlGrid< 3, double, tnlHost, int >, double, int >,
                                                             tnlExactDifference< 3 >,
                                                             tnlExpBumpFunction< 3, double >,
                                                             MeshSize,
                                                             verbose > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#endif
   
#else
   return EXIT_FAILURE;
#endif
}

