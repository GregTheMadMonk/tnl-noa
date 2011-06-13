/***************************************************************************
                          tnl-unit-tests.h  -  description
                             -------------------
    begin                : Nov 21, 2009
    copyright            : (C) 2009 by Tomas Oberhuber
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

#include <debug/tnlDebug.h>
#include "core/tnlFileTester.h"
#include "core/tnlRealTester.h"
#include "core/tnlVectorTester.h"
#include "core/tnlLongVectorHostTester.h"
#include "core/tnlArrayTester.h"
#include "core/tnlGridTester.h"
#include "core/tnlSharedMemoryTester.h"
#include "core/tnlCommunicatorTester.h"
#include "matrix/tnlCSRMatrixTester.h"
#include "matrix/tnlRgCSRMatrixTester.h"
#include "matrix/tnlAdaptiveRgCSRMatrixTester.h"
#include "matrix/tnlEllpackMatrixTester.h"
#include "diff/tnlMPIMeshTester.h"

#include <iostream>

using namespace std;

int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "" );

   CppUnit :: TextTestRunner runner;

   runner. addTest( tnlFileTester :: suite() );

   runner. addTest( tnlRealTester< float > :: suite() );
   runner. addTest( tnlRealTester< double > :: suite() );

   runner. addTest( tnlVectorTester< 1, int > :: suite() );
   runner. addTest( tnlVectorTester< 2, int > :: suite() );
   runner. addTest( tnlVectorTester< 3, int > :: suite() );
   runner. addTest( tnlVectorTester< 4, int > :: suite() );
   runner. addTest( tnlVectorTester< 1, float > :: suite() );
   runner. addTest( tnlVectorTester< 2, float > :: suite() );
   runner. addTest( tnlVectorTester< 3, float > :: suite() );
   runner. addTest( tnlVectorTester< 4, float > :: suite() );
   runner. addTest( tnlVectorTester< 1, double > :: suite() );
   runner. addTest( tnlVectorTester< 2, double > :: suite() );
   runner. addTest( tnlVectorTester< 3, double > :: suite() );
   runner. addTest( tnlVectorTester< 4, double > :: suite() );


   runner. addTest( tnlLongVectorHostTester< int > :: suite() );
   runner. addTest( tnlLongVectorHostTester< float > :: suite() );
   runner. addTest( tnlLongVectorHostTester< double > :: suite() );

   runner. addTest( tnlArrayTester< int, tnlHost, int > :: suite() );
   runner. addTest( tnlArrayTester< float, tnlHost, int > :: suite() );
   runner. addTest( tnlArrayTester< double, tnlHost, int > :: suite() );

   /*runner. addTest( tnlCSRMatrixTester< float > :: suite() );
   runner. addTest( tnlCSRMatrixTester< double > :: suite() );*/

   runner. addTest( tnlRgCSRMatrixTester< float > :: suite() );
   runner. addTest( tnlRgCSRMatrixTester< double > :: suite() );

   runner. addTest( tnlEllpackMatrixTester< float > :: suite() );
   runner. addTest( tnlEllpackMatrixTester< double > :: suite() );

   //runner. addTest( tnlAdaptiveRgCSRMatrixTester< float > :: suite() );
   runner. addTest( tnlAdaptiveRgCSRMatrixTester< double, tnlHost > :: suite() );

   runner. addTest( tnlMPIMeshTester< float > :: suite() );

   //runner. addTest( tnlSharedMemoryTester< tnlHost > :: suite() );

   //runner. addTest( tnlCommunicatorTester< tnlHost > :: suite() );


#ifdef HAVE_CUDA
   runner. addTest( tnlFileTester :: suite() );

   /*runner.addTest( tnlLongVectorCUDATester< int > :: suite() );
   runner.addTest( tnlLongVectorCUDATester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlLongVectorCUDATester< double > :: suite() );

   runner.addTest( tnlFieldCUDA2DTester< int > :: suite() );
   runner.addTest( tnlFieldCUDA2DTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlFieldCUDA2DTester< double > :: suite() );

   runner.addTest( tnlGridCUDA2DTester< int > :: suite() );
   runner.addTest( tnlGridCUDA2DTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlGridCUDA2DTester< double > :: suite() );

   runner.addTest( tnlCUDAKernelsTester< int > :: suite() );
   runner.addTest( tnlCUDAKernelsTester< float > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlCUDAKernelsTester< double > :: suite() );

   runner.addTest( tnlMersonSolverTester< float, tnlCuda, int > :: suite() );
   if( CUDA_ARCH == 13 )
      runner.addTest( tnlMersonSolverTester< double, tnlCuda, int > :: suite() );*/

   runner. addTest( tnlAdaptiveRgCSRMatrixTester< double, tnlCuda > :: suite() );
#endif


   runner.run();
   return 0;
}
