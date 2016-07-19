/***************************************************************************
                          tnlEllpackIndexMultimapTest.cpp  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include <TNL/core/multimaps/tnlEllpackIndexMultimap.h>
#include "tnlIndexMultimapTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlIndexMultimapTester< tnlEllpackIndexMultimap< int, tnlHost > > >() ||
       ! tnlUnitTestStarter :: run< tnlIndexMultimapTester< tnlEllpackIndexMultimap< long int, tnlHost > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
