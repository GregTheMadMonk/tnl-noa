/***************************************************************************
                          tnl-unit-tests.cpp  -  description
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

#include <core/tnlLongVectorCUDATester.h>

using namespace std;

__global__ void setNumber( float* A, float c )
{
   int i = threadIdx. x;
   A[ i ] = 1.0;
}

int main( int argc, char* argv[] )
{
   int size = 100;
   float *h_a, *d_a;
   cudaMalloc( ( void** )&d_a, size * sizeof( float ) );
   h_a = ( float* ) malloc( size * sizeof( float ) );
   setNumber<<< 1, size >>>( d_a, 1.0 );
   cudaMemcpy( h_a, d_a, size * sizeof( float ), cudaMemcpyDeviceToHost );
   for( int i = 0; i < size; i ++ )
      cout << h_a[ i ] << "-";
}


/*int main( int argc, char* argv[] )
{
   CppUnit::TextUi::TestRunner runner;
   runner.addTest( tnlLongVectorCUDATester< float > :: suite() );
   //runner.addTest( ComplexNumberTest::suite() );
   runner.run();
   return 0;
}*/





#ifdef UNDEF

#include <stdlib.h>
#include <core/tnlTester.h>
#include <core/tnlStringTester.h>
#include <core/tnlObjectTester.h>

int main( int argc, char* argv[] )
{
   tnlTester tester;

   /* Testing tnlString
    *
    */
   tnlStringTester string_tester;
   string_tester. Test( tester );

   /* Testing tnlObject
    *
    */
   tnlObjectTester tnl_object_tester;
   tnl_object_tester. Test( tester );

   tester. PrintStatistics();

   return EXIT_SUCCESS;
}
#endif
