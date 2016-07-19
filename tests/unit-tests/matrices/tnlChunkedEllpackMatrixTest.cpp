/***************************************************************************
                          tnlChunkedEllpackMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 13, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <TNL/matrices/tnlChunkedEllpackMatrix.h>
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
