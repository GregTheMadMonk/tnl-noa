/***************************************************************************
                          Devices::CudaVectorOperationsTest.cu  -  description
                             -------------------
    begin                : Mar 31, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
 
#include "tnlVectorOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
 
int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter :: run< VectorOperationsTester< double, Devices::Cuda > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
#else
   return EXIT_FAILURE;
#endif
}
