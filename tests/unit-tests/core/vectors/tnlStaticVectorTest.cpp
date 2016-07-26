/***************************************************************************
                          StaticVectorTest.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include "tnlStaticVectorTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< StaticVectorTester< 1, double > >() ||
       ! tnlUnitTestStarter::run< StaticVectorTester< 2, double > >() ||
       ! tnlUnitTestStarter::run< StaticVectorTester< 3, double > >() ||
       ! tnlUnitTestStarter::run< StaticVectorTester< 4, double > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


