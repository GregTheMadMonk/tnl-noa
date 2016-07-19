/***************************************************************************
                          tnlTridiagonalMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include "tnlTridiagonalMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlTridiagonalMatrixTester< float, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlTridiagonalMatrixTester< double, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlTridiagonalMatrixTester< float, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlTridiagonalMatrixTester< double, tnlCuda, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
