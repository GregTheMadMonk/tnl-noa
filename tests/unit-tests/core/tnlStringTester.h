/***************************************************************************
                          tnlStringTester.h -  description
                             -------------------
    begin                : Oct 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSTRINGTESTER_H_
#define TNLSTRINGTESTER_H_

#include <string.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlString.h>
#include <core/tnlFile.h>

class tnlStringTester : public CppUnit :: TestCase
{
   public:
   tnlStringTester(){};

   virtual
   ~tnlStringTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlStringTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testBasicConstructor",
                               & tnlStringTester :: testBasicConstructor )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testConstructorWithChar",
                               & tnlStringTester :: testConstructorWithChar )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testCopyConstructor",
                               & tnlStringTester :: testCopyConstructor )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testConstructorWithNumber",
                               & tnlStringTester :: testConstructorWithNumber )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testSetString",
                               & tnlStringTester :: testSetString )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testIndexingOperator",
                               & tnlStringTester :: testIndexingOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testAssignmentOperator",
                               & tnlStringTester :: testAssignmentOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testAdditionAssignmentOperator",
                               & tnlStringTester :: testAdditionAssignmentOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStringTester >(
                               "testSave",
                               & tnlStringTester :: testSave )
                              );
       return suiteOfTests;
   }

   void testBasicConstructor()
   {
      tnlString str;
      CPPUNIT_ASSERT( strcmp( str. getString(), "" ) == 0 );
   }

   void testConstructorWithChar()
   {
      tnlString str1( "string1" );
      tnlString str2( "xxxstring2", 3 );
      tnlString str3( "string3xxx", 0, 3 );
      tnlString str4( "xxxstring4xxx", 3, 3 );

      CPPUNIT_ASSERT( strcmp( str1. getString(), "string1" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str2. getString(), "string2" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str3. getString(), "string3" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str4. getString(), "string4" ) == 0 );
   }

   void testCopyConstructor()
   {
      tnlString string( "string1" );
      tnlString emptyString( "" );
      tnlString string2( string );
      tnlString emptyString2( emptyString );

      CPPUNIT_ASSERT( strcmp( string2. getString(), "string1" ) == 0 );
      CPPUNIT_ASSERT( strcmp( emptyString2. getString(), "" ) == 0 );
   }

   void testConstructorWithNumber()
   {
      tnlString string1( 10 );
      tnlString string2( -5 );

      CPPUNIT_ASSERT( strcmp( string1. getString(), "10" ) == 0 );
      CPPUNIT_ASSERT( strcmp( string2. getString(), "-5" ) == 0 );
   }

   void testSetString()
   {
      tnlString str1, str2, str3, str4;

      str1. setString( "string1" );
      str2. setString( "xxxstring2", 3 );
      str3. setString( "string3xxx", 0, 3 );
      str4. setString( "xxxstring4xxx", 3, 3 );

      CPPUNIT_ASSERT( strcmp( str1. getString(), "string1" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str2. getString(), "string2" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str3. getString(), "string3" ) == 0 );
      CPPUNIT_ASSERT( strcmp( str4. getString(), "string4" ) == 0 );
   }

   void testIndexingOperator()
   {
      tnlString str( "1234567890" );
      CPPUNIT_ASSERT( str[ 0 ] == '1' );
      CPPUNIT_ASSERT( str[ 1 ] == '2' );
      CPPUNIT_ASSERT( str[ 2 ] == '3' );
      CPPUNIT_ASSERT( str[ 3 ] == '4' );
      CPPUNIT_ASSERT( str[ 4 ] == '5' );
      CPPUNIT_ASSERT( str[ 5 ] == '6' );
      CPPUNIT_ASSERT( str[ 6 ] == '7' );
      CPPUNIT_ASSERT( str[ 7 ] == '8' );
      CPPUNIT_ASSERT( str[ 8 ] == '9' );
      CPPUNIT_ASSERT( str[ 9 ] == '0' );
   }

   void testAssignmentOperator()
   {
      tnlString string1( "string" );
      tnlString string2;
      string2 = string1;

      CPPUNIT_ASSERT( strcmp( string2. getString(), "string" ) == 0 );
   }

   void testAdditionAssignmentOperator()
   {
      tnlString string1( "string" );
      tnlString string2;
      string2 = string1;
      string2 += "string2";

      CPPUNIT_ASSERT( strcmp( string2. getString(), "stringstring2" ) == 0 );
   }


   void testSave()
   {
      tnlString str1( "testing-string" );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode, tnlCompressionBzip2 );
      str1. save( file );
   };

};

#endif /* TNLSTRINGTESTER_H_ */
