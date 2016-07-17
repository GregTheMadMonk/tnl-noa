/***************************************************************************
                          tnlStaticArrayTest.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "tnlStaticArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlStaticArrayTester< 1, int > >() ||
       ! tnlUnitTestStarter::run< tnlStaticArrayTester< 2, int > >() ||
       ! tnlUnitTestStarter::run< tnlStaticArrayTester< 3, int > >() ||
       ! tnlUnitTestStarter::run< tnlStaticArrayTester< 4, int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


