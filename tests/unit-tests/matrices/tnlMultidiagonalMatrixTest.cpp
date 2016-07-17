/***************************************************************************
                          tnlMultidiagonalMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 4, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "tnlMultidiagonalMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlMultidiagonalMatrixTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultidiagonalMatrixTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultidiagonalMatrixTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultidiagonalMatrixTester< double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}

