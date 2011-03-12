/***************************************************************************
                          tnl-cuda-unit-tests.cu  -  description
                             -------------------
    begin                : Oct 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#include <cppunit/ui/text/TestRunner.h>

#include <core/tnlFileTester.h>
#include <core/tnlLongVectorCUDATester.h>
#include <core/tnlCUDAKernelsTester.h>
#include <solver/tnlMersonSolverTester.h>

#include <iostream>

using namespace std;

int main( int argc, char* argv[] )
{
   CppUnit :: TextTestRunner runner;
   
#ifdef HAVE_CUDA
   runner. addTest( tnlFileTester :: suite() );

   runner.addTest( tnlLongVectorCUDATester< int > :: suite() );
   runner.addTest( tnlLongVectorCUDATester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlLongVectorCUDATester< double > :: suite() );
   
   /*runner.addTest( tnlFieldCUDA2DTester< int > :: suite() );
   runner.addTest( tnlFieldCUDA2DTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlFieldCUDA2DTester< double > :: suite() );
   
   runner.addTest( tnlGridCUDA2DTester< int > :: suite() );
   runner.addTest( tnlGridCUDA2DTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlGridCUDA2DTester< double > :: suite() );
   */
   /*runner.addTest( tnlCUDAKernelsTester< int > :: suite() );
   runner.addTest( tnlCUDAKernelsTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlCUDAKernelsTester< double > :: suite() );*/

   runner.addTest( tnlMersonSolverTester< float, tnlCuda, int > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlMersonSolverTester< double, tnlCuda, int > :: suite() );
#endif
  
   runner.run();
   return 0;
}
