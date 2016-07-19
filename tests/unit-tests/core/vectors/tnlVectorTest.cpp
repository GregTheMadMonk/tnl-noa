/***************************************************************************
                          tnlVectorTest.cpp  -  description
                             -------------------
    begin                : Jul 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include "tnlVectorTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlVectorTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlVectorTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlVectorTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlVectorTester< double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


