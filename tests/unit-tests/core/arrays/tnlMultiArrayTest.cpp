/***************************************************************************
                          MultiArrayTest.cpp  -  description
                             -------------------
    begin                : Feb 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlMultiArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< MultiArrayTester< 1, char, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, long int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, char, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, long int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, double, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, char, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, long int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, char, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, long int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, double, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, char, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, long int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, char, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, long int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, double, Devices::Host, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
