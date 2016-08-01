/***************************************************************************
                          ChunkedEllpackMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/ChunkedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "tnlChunkedEllpackMatrixTestSetup.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< float, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< double, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< float, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< double, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< float, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< ChunkedEllpackMatrix< double, Devices::Cuda, int >, ChunkedEllpackMatrixTestSetup< 2, 16 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
