/***************************************************************************
                          memory-operations-test.cu  -  description
                             -------------------
    begin                : Mar 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <tnlConfig.h>
 
#ifdef HAVE_CPPUNIT

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <iostream>
#include "tnlCudaMemoryOperationsTester.h"
 
int main( int argc, char* argv[] )
{
   CppUnit :: TextTestRunner runner;
   runner. addTest( tnlCudaMemoryOperationsTester :: suite() );
   //runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cout));
   if( ! runner.run() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif