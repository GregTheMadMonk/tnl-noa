/***************************************************************************
                          ArrayTest.cpp  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include "tnlArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< ArrayTester< char, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< char, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
