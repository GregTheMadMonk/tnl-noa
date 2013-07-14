/***************************************************************************
                          tnlArrayTest.cpp  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "tnlArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlArrayTester< char, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< long int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< long double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< char, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< long int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< double, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlArrayTester< long double, tnlHost, long int > >() )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
