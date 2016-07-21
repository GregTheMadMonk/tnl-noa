/***************************************************************************
                          tnlVectorOperationsTest.cpp  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnlVectorOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
#include <TNL/core/tnlHost.h>

using namespace TNL;

int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlVectorOperationsTester< double, tnlHost > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}



