/***************************************************************************
                          ArrayTest.cpp  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< ArrayTester< char, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< char, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< long int, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< ArrayTester< double, Devices::Host, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
