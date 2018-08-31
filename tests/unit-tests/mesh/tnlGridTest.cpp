/***************************************************************************
                          GridTest.cpp  -  description
                             -------------------
    begin                : Jul 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#ifndef HAVE_NOT_CXX11
#include "tnlGridTester.h"
#endif
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifndef HAVE_NOT_CXX11
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< GridTester< 1, double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< GridTester< 2, double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< GridTester< 3, double, Devices::Host, int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
#endif
}




