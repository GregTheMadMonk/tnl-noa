/***************************************************************************
                          ChunkedEllpackTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/ChunkedEllpack.h>
#include <cstdlib>

#include "SparseMatrixTester.h"
#include "tnlChunkedEllpackMatrixTestSetup.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< float, Devices::Cuda, int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< double, Devices::Cuda, int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< float, Devices::Cuda, int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< double, Devices::Cuda, int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< float, Devices::Cuda, int >, ChunkedEllpackTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< ChunkedEllpack< double, Devices::Cuda, int >, ChunkedEllpackTestSetup< 2, 16 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
