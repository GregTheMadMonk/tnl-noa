/***************************************************************************
                          tnlArrayOperationsTest.cpp  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnlArrayOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
#include <core/tnlHost.h>

int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlArrayOperationsTester< int, tnlHost > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}




