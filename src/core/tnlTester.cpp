/***************************************************************************
                          tnlTester.cpp  -  description
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

#include <iostream>
#include <cstdlib>
#include "tnlTester.h"

using namespace std;

//--------------------------------------------------------------------------
tnlTester :: tnlTester()
   : tests_passed_counter( 0 ),
     tests_failed_counter( 0 ),
     tests_not_implemented_counter( 0 )
{

}
//--------------------------------------------------------------------------
void tnlTester :: StartNewUnit( const char* unit_name )
{
   if( current_unit )
   {
      cerr << "There is already one unit being tested now: " << current_unit << endl;
      cerr << "Please fix the testing program." << endl;
      cerr << "Exiting the test." << endl;
      exit( EXIT_FAILURE );
   }
   cout << "Starting testing unit : " << unit_name << endl;
   current_unit. SetString( unit_name );
}
//--------------------------------------------------------------------------
void tnlTester :: FinishUnit()
{
   if( ! current_unit )
   {
      cerr << "Unable to finish unit testing since no unit test has been started." << endl;
      cerr << "Please fix the testing program." << endl;
      cerr << "Exiting the test." << endl;
      exit( EXIT_FAILURE );
   }
   cout << "Finishing testing unit : " << current_unit << endl;
   current_unit. SetString( "" );
}
//--------------------------------------------------------------------------
void tnlTester :: StartNewTest( const char* test_description )
{
   cout << "Checking " << test_description;
   tests_stack. Push( tnlString( test_description ) );
}
//--------------------------------------------------------------------------
void tnlTester :: FinishTest( tnlTestResult test_result )
{
   if( tests_stack. IsEmpty() )
   {
      cerr << "Unable to finish test since the tests stack is empty." << endl;
      cerr << "Please fix the testing program." << endl;
      cerr << "Exiting the test." << endl;
      exit( EXIT_FAILURE );
   }
   switch( test_result )
   {
      case tnlTestPASS:
         cout << ".... [ PASSED ] " << endl;
         tests_passed_counter ++;
         break;
      case tnlTestFAIL:
         cout << ".... [ FAILED ] " << endl;
         tests_failed_counter ++;
         break;
      case tnlTestNOT_IMPLEMENTED:
         cout << "... [ NOT IMPLEMENTED ] " << endl;
         tests_not_implemented_counter ++;
         break;
      default:
         cerr << "Unknown test result !" << endl;
         cerr << "Please check the testing code." << endl;
         cerr << "Exiting test." << endl;
         exit( EXIT_FAILURE );
   }
   tnlString test_description;
   tests_stack. Pop( test_description );
}
//--------------------------------------------------------------------------
int tnlTester :: GetTestPASSNumber()
{
   return tests_passed_counter;
}
//--------------------------------------------------------------------------
int tnlTester :: GetTestFAILNumber()
{
   return tests_failed_counter;
}
//--------------------------------------------------------------------------
int tnlTester :: GetTestNOTIMPLEMENTDNumber()
{
   return tests_not_implemented_counter;
}
//--------------------------------------------------------------------------
void tnlTester :: PrintStatistics()
{
   cout << "Number of succesfuly PASSED tests: " << GetTestPASSNumber() << endl;
   cout << "Number of FAILED tests: " << GetTestFAILNumber() << endl;
   cout << "Number of NON-IMPLEMENTED tests: " << GetTestNOTIMPLEMENTDNumber() << endl;
   cout << "Total number of tests: " << GetTestNOTIMPLEMENTDNumber()
            + GetTestFAILNumber() + GetTestPASSNumber() << endl;
}

