/***************************************************************************
                          tnlCSRMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "matrices/tnlCSRMatrix.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlCSRMatrix< float, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlCSRMatrix< double, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlCSRMatrix< float, tnlHost, long int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlCSRMatrix< double, tnlHost, long int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
