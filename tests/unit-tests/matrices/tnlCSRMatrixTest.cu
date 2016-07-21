/***************************************************************************
                          tnlCSRMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/core/tnlHost.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlCSRMatrix< float, tnlCuda, int > > >() ||
       ! tnlUnitTestStarter::run< tnlSparseMatrixTester< tnlCSRMatrix< double, tnlCuda, int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}

