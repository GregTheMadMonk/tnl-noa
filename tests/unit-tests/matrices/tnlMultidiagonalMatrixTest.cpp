/***************************************************************************
                          MultidiagonalMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 4, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlMultidiagonalMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< MultidiagonalMatrixTester< float, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultidiagonalMatrixTester< double, Devices::Host, int > >() ||
       ! tnlUnitTestStarter :: run< MultidiagonalMatrixTester< float, Devices::Host, long int > >() ||
       ! tnlUnitTestStarter :: run< MultidiagonalMatrixTester< double, Devices::Host, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}

