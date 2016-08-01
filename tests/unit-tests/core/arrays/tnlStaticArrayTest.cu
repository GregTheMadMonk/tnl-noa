/***************************************************************************
                          StaticArrayTest.cu  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlStaticArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< StaticArrayTester< 1, int > >() ||
       ! tnlUnitTestStarter::run< StaticArrayTester< 2, int > >() ||
       ! tnlUnitTestStarter::run< StaticArrayTester< 3, int > >() ||
       ! tnlUnitTestStarter::run< StaticArrayTester< 4, int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
