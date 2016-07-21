/***************************************************************************
                          tnlSlicedEllpackMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlCuda.h>
#include <TNL/matrices/tnlSlicedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
  if( ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlCuda, int, 32 > > >() ||
      ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlCuda, int, 32 > > >() ||
      ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlCuda, int, 4 > > >() ||
      ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlCuda, int, 4 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
