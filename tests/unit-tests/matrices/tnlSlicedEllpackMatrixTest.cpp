/***************************************************************************
                          tnlSlicedEllpackMatrixTest.cpp  -  description
                             -------------------
    begin                : Dec 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlHost, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlHost, int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlHost, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlHost, long int, 32 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlHost, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlHost, int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< float, tnlHost, long int, 4 > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlSlicedEllpackMatrix< double, tnlHost, long int, 4 > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}


