/***************************************************************************
                          tnlChunkedEllpackMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
