/***************************************************************************
                          VectorTest.cpp  -  description
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
   if( ! tnlUnitTestStarter :: run< VectorTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< VectorTester< double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


