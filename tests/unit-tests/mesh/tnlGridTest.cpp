/***************************************************************************
                          tnlGridTest.cpp  -  description
                             -------------------
    begin                : Jul 28, 2014
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

#ifndef HAVE_NOT_CXX11
#include "tnlGridTester.h"
#endif
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifndef HAVE_NOT_CXX11
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlGridTester< 1, double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlGridTester< 2, double, tnlHost, int > >() ||
       ! tnlUnitTestStarter :: run< tnlGridTester< 3, double, tnlHost, int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
#endif
}




