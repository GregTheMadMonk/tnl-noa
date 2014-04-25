/***************************************************************************
                          tnlEllpackTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
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
#include <core/tnlCuda.h>
#include <matrices/tnlEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlEllpackMatrix< float, tnlCuda, int > > >() ||
       ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlEllpackMatrix< double, tnlCuda, int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}