/***************************************************************************
                          tnlStringTester.h -  description
                             -------------------
    begin                : Oct 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSTRINGTESTER_H_
#define TNLSTRINGTESTER_H_

#ifdef HAVE_CPPUNIT

#include <string.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlString.h>
#include <core/tnlFile.h>

using namespace TNL;

class tnlStringTester : public CppUnit :: TestCase
{
   public:

   typedef tnlStringTester StringTester;
   typedef CppUnit :: TestCaller< StringTester > TestCaller;

   tnlStringTester(){};

   virtual
   ~tnlStringTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlStringTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testBasicConstructor", &tnlStringTester :: testBasicConstructor ) );
      suiteOfTests -> addTest( new TestCaller( "testConstructorWithChar", &tnlStringTester :: testConstructorWithChar ) );
      suiteOfTests -> addTest( new TestCaller( "testCopyConstructor", &tnlStringTester :: testCopyConstructor ) );
      suiteOfTests -> addTest( new TestCaller( "testConstructorWithNumber", &tnlStringTester :: testConstructorWithNumber ) );
      suiteOfTests -> addTest( new TestCaller( "testSetString", &tnlStringTester :: testSetString ) );
      suiteOfTests -> addTest( new TestCaller( "testIndexingOperator", &tnlStringTester :: testIndexingOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testAssignmentOperator", &tnlStringTester :: testAssignmentOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testAdditionAssignmentOperator", &tnlStringTester :: testAdditionAssignmentOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testSave", &tnlStringTester :: testSave ) );
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
      file. open( "test-file.tnl", tnlWriteMode );
      str1. save( file );
   };

};

#endif /* HAVE_CPPUNIT */

#endif /* TNLSTRINGTESTER_H_ */
