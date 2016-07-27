/***************************************************************************
                          tnlMeshTest.cpp  -  description
                             -------------------
    begin                : Feb 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#ifndef HAVE_NOT_CXX11
#include "tnlMeshTester.h"
#endif
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifndef HAVE_NOT_CXX11
#ifdef HAVE_CPPUNIT
   tnlMeshTester< double, Devices::Host, long int > t;
   //t.regularMeshOfHexahedronsTest();
   if( ! tnlUnitTestStarter :: run< tnlMeshTester< double, Devices::Host, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
#endif
}


