/***************************************************************************
                          EllpackIndexMultimapTest.cpp  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include <TNL/Experimental/Multimaps/EllpackIndexMultimap.h>
#include "tnlIndexMultimapTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlIndexMultimapTester< EllpackIndexMultimap< int, Devices::Host > > >() ||
       ! tnlUnitTestStarter :: run< tnlIndexMultimapTester< EllpackIndexMultimap< long int, Devices::Host > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
