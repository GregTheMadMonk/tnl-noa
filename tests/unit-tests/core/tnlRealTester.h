/***************************************************************************
                          tnlRealTester.h  -  description
                             -------------------
    begin                : Jun 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLREALTESTER_H_
#define TNLREALTESTER_H_

/*
 *
 */
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlReal.h>

template< class T > class tnlRealTester : public CppUnit :: TestCase
{
   public:
   tnlRealTester(){};

   virtual
   ~tnlRealTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlRealTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRealTester< T > >(
                               "testComparisonOperators",
                               & tnlRealTester< T > :: testComparisonOperators ) );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRealTester< T > >(
                               "testOperatorPlus",
                               & tnlRealTester< T > :: testOperatorPlus ) );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRealTester< T > >(
                         "testOperatorDivide",
                         & tnlRealTester< T > :: testOperatorDivide ) );


      return suiteOfTests;
   }

   void testComparisonOperators()
   {
      tnlReal< T > a( 1.5 ), b( 2.5 ), c( 1.5 );
      CPPUNIT_ASSERT( a == c );
      CPPUNIT_ASSERT( a < b );
      CPPUNIT_ASSERT( a <= b );
      CPPUNIT_ASSERT( a <= c );
      CPPUNIT_ASSERT( b > a );
      CPPUNIT_ASSERT( b >= a );
      CPPUNIT_ASSERT( a != b );

   };

   void testOperatorPlus()
   {
      T a( 1.5 ), b( 2.5 );
      tnlReal< T > ta( 1.5 ), tb( 2.5 );
      tnlReal< T > result = ta + tb;
      //CPPUNIT_ASSERT( result. Data() == a + b );
      //CPPUNIT_ASSERT( result == a + b );
   };

   void testOperatorDivide()
   {
      T a( 2.0 );
      tnlReal< T > ta( 2.0 );

      T b = 1.0 / a;
      tnlReal< T > tb = 1.0 / ta;

      CPPUNIT_ASSERT( b == tb );

      int ia( 2 );

      const tnlReal< T > tbi = 1.0 / ( T ) ia;

      //cerr << tbi << endl;

      CPPUNIT_ASSERT( b == tbi );
   };
};

#endif /* TNLREALTESTER_H_ */
