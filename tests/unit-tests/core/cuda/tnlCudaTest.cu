/***************************************************************************
                          Devices::CudaTest.cu  -  description
                             -------------------
    begin                : Mar 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
 
#include "tnlCudaTester.h"
#include "../../tnlUnitTestStarter.h"
 
int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlCudaTester >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
 
