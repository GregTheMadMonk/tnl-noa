/***************************************************************************
                          SlicedEllpackMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/SlicedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< float, Devices::Host, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< double, Devices::Host, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< float, Devices::Host, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< double, Devices::Host, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< float, Devices::Host, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< double, Devices::Host, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< float, Devices::Host, long int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::SlicedEllpackMatrix< double, Devices::Host, long int, 4 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


