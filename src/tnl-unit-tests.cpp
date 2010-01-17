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
#include <core/tnlFieldCUDA2DTester.h>
#include <core/tnlCUDAKernelsTester.h>
#include <diff/tnlGridCUDA2DTester.h>

#include <iostream>

using namespace std;

int main( int argc, char* argv[] )
{
   CppUnit::TextUi::TestRunner runner;
   
   runner.addTest( tnlLongVectorCUDATester< int > :: suite() );
   runner.addTest( tnlLongVectorCUDATester< float > :: suite() );
   runner.addTest( tnlLongVectorCUDATester< double > :: suite() );
   
   runner.addTest( tnlFieldCUDA2DTester< int > :: suite() );
   runner.addTest( tnlFieldCUDA2DTester< float > :: suite() );
   runner.addTest( tnlFieldCUDA2DTester< double > :: suite() );
   
   runner.addTest( tnlGridCUDA2DTester< int > :: suite() );
   runner.addTest( tnlGridCUDA2DTester< float > :: suite() );
   runner.addTest( tnlGridCUDA2DTester< double > :: suite() );
   
   runner.addTest( tnlCUDAKernelsTester< int > :: suite() );
   //runner.addTest( tnlCUDAKernelsTester< float > :: suite() );
   //runner.addTest( tnlCUDAKernelsTester< double > :: suite() );
   
   runner.run();
   return 0;
}





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
