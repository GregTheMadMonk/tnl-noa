/***************************************************************************
                          EllpackTest.cpp  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlSparseMatrixTester.h"
#include <TNL/Matrices/Ellpack.h>
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< SparseTester< Matrices::Ellpack< float, Devices::Host, int > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::Ellpack< double, Devices::Host, int > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::Ellpack< float, Devices::Host, long int > > >() ||
       ! tnlUnitTestStarter :: run< SparseTester< Matrices::Ellpack< double, Devices::Host, long int > > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
