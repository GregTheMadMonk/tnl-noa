/***************************************************************************
                          tnlMultiArrayTest.cpp  -  description
                             -------------------
    begin                : Feb 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#include "tnlMultiArrayTester.h"
#include "../../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, char, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, long int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, char, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, long int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 1, double, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, char, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, long int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, char, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, long int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 2, double, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, char, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, long int, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, float, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, char, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, long int, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, float, tnlHost, long int > >() ||
       ! tnlUnitTestStarter :: run< tnlMultiArrayTester< 3, double, tnlHost, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
