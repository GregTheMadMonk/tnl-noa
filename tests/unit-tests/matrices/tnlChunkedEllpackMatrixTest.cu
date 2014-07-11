/***************************************************************************
                          tnlChunkedEllpackMatrixTest.cu  -  description
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
#include <matrices/tnlChunkedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "tnlChunkedEllpackMatrixTestSetup.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< float, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlChunkedEllpackMatrix< double, tnlCuda, int >, tnlChunkedEllpackMatrixTestSetup< 2, 16 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
