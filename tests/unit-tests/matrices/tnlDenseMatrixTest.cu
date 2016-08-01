/***************************************************************************
                          DenseMatrixTest.cu  -  description
                             -------------------
    begin                : Jan 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "tnlDenseMatrixTester.h"
#include "../tnlUnitTestStarter.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< DenseMatrixTester< float, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< DenseMatrixTester< double, Devices::Cuda, int > >() ||
       ! tnlUnitTestStarter :: run< DenseMatrixTester< float, Devices::Cuda, long int > >() ||
       ! tnlUnitTestStarter :: run< DenseMatrixTester< double, Devices::Cuda, long int > >()
       )
     return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}

