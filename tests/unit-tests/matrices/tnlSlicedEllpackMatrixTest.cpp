/***************************************************************************
                          SlicedEllpackTest.cpp  -  description
                             -------------------
    begin                : Dec 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/SlicedEllpack.h>
#include <cstdlib>

#include "SparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< float, Devices::Host, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< double, Devices::Host, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< float, Devices::Host, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< double, Devices::Host, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< float, Devices::Host, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< double, Devices::Host, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< float, Devices::Host, long int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::SlicedEllpack< double, Devices::Host, long int, 4 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


