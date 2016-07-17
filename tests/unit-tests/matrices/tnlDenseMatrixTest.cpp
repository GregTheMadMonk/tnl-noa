/***************************************************************************
                          tnlDenseMatrixTest.cpp  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "tnlDenseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlDenseMatrixTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlDenseMatrixTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlDenseMatrixTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlDenseMatrixTester< double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


