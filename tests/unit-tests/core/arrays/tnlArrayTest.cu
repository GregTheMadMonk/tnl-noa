/***************************************************************************
                          ArrayTest.cu  -  description
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
   if( //! tnlUnitTestStarter :: run< ArrayTester< char, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< char, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, tnlCuda, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
