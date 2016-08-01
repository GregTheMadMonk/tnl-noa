/***************************************************************************
                          SlicedEllpackMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SlicedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
  if( ! tnlUnitTestStarter::run< SparseMatrixTester< SlicedEllpackMatrix< float, Devices::Cuda, int, 32 > > >() ||
      ! tnlUnitTestStarter::run< SparseMatrixTester< SlicedEllpackMatrix< double, Devices::Cuda, int, 32 > > >() ||
      ! tnlUnitTestStarter::run< SparseMatrixTester< SlicedEllpackMatrix< float, Devices::Cuda, int, 4 > > >() ||
      ! tnlUnitTestStarter::run< SparseMatrixTester< SlicedEllpackMatrix< double, Devices::Cuda, int, 4 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
