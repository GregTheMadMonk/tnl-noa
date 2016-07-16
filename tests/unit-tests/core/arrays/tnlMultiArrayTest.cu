/***************************************************************************
                          tnlMultiArrayTest.cu  -  description
                             -------------------
    begin                : Feb 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlCuda.h>
#include <cstdlib>

#include "tnlMultiArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, char, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, long int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, float, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, double, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, char, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, long int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, float, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, double, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, char, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, long int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, float, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, double, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, char, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, long int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, float, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, double, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, char, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, long int, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, float, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, double, tnlCuda, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, char, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, int, tnlCuda, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, long int, tnlCuda, long int > >() ||
       //! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, float, tnlCuda, long int > >()
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, double, tnlCuda, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
