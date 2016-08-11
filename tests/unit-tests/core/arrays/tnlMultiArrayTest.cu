/***************************************************************************
                          MultiArrayTest.cu  -  description
                             -------------------
    begin                : Feb 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Cuda.h>
#include <cstdlib>

#include "tnlMultiArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< MultiArrayTester< 1, char, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, long int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, float, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, double, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, char, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, int, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, long int, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, float, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 1, double, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, char, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, long int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, float, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, double, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, char, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, int, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, long int, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, float, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 2, double, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, char, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, long int, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, float, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, double, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, char, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, int, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, long int, Devices::Cuda, long int > >() ||
       //! tnlUnitTestStarter :: run< MultiArrayTester< 3, float, Devices::Cuda, long int > >()
       ! tnlUnitTestStarter :: run< MultiArrayTester< 3, double, Devices::Cuda, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
