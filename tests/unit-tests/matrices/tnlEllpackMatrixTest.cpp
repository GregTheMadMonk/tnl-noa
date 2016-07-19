/***************************************************************************
                          tnlEllpackTest.cpp  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include <TNL/matrices/tnlEllpackMatrix.h>
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< float, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< double, tnlHost, int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< float, tnlHost, long int > > >() ||
       ! tnlUnitTestStarter :: run< tnlSparseMatrixTester< tnlEllpackMatrix< double, tnlHost, long int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
