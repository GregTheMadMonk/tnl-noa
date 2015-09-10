/***************************************************************************
                          tnlEllpackGraphTest.cpp  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#include "graphs/tnlEllpackGraph.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< float, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< double, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< float, tnlHost, long int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< double, tnlHost, long int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
