/***************************************************************************
                          tnlChunkedEllpackMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 13, 2013
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
#include <matrices/tnlChunkedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "tnlChunkedEllpackMatrixTestSetup.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlHost, long int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
