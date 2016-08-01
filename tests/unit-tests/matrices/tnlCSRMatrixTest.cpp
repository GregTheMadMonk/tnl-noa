/***************************************************************************
                          CSRMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include <TNL/Matrices/CSRMatrix.h>
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::CSRMatrix< float, Devices::Host, int > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::CSRMatrix< double, Devices::Host, int > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::CSRMatrix< float, Devices::Host, long int > > >() ||
       ! tnlUnitTestStarter :: run< SparseMatrixTester< Matrices::CSRMatrix< double, Devices::Host, long int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
