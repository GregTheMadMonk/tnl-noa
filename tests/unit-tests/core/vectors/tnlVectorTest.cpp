/***************************************************************************
                          VectorTest.cpp  -  description
                             -------------------
    begin                : Jul 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlVectorTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< VectorTester< float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< double, Devices::Host, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


