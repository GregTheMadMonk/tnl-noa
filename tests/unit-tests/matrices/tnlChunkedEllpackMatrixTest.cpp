/***************************************************************************
                          ChunkedEllpackTest.cpp  -  description
                             -------------------
    begin                : Dec 13, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/ChunkedEllpack.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "tnlChunkedEllpackMatrixTestSetup.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, long int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, long int >, ChunkedEllpackTestSetup< 4, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, long int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, long int >, ChunkedEllpackTestSetup< 16, 2 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, int >, ChunkedEllpackTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, int >, ChunkedEllpackTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< float, Devices::Host, long int >, ChunkedEllpackTestSetup< 2, 16 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::ChunkedEllpack< double, Devices::Host, long int >, ChunkedEllpackTestSetup< 2, 16 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
