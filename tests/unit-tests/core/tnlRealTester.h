/***************************************************************************
                          RealTester.h  -  description
                             -------------------
    begin                : Jun 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef RealTESTER_H_
#define RealTESTER_H_

/*
 *
 */
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Experimental/Arithmetics/Real.h>

template< class T > class RealTester : public CppUnit :: TestCase
{
   public:
   RealTester(){};

   virtual
   ~RealTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "RealTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< RealTester< T > >(
                               "testComparisonOperators",
                               & RealTester< T > :: testComparisonOperators ) );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< RealTester< T > >(
                               "testOperatorPlus",
                               & RealTester< T > :: testOperatorPlus ) );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< RealTester< T > >(
                         "testOperatorDivide",
                         & RealTester< T > :: testOperatorDivide ) );


      return suiteOfTests;
   }

   void testComparisonOperators()
   {
      Real< T > a( 1.5 ), b( 2.5 ), c( 1.5 );
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
      Real< T > ta( 1.5 ), tb( 2.5 );
      Real< T > result = ta + tb;
      //CPPUNIT_ASSERT( result. Data() == a + b );
      //CPPUNIT_ASSERT( result == a + b );
   };

   void testOperatorDivide()
   {
      T a( 2.0 );
      Real< T > ta( 2.0 );

      T b = 1.0 / a;
      Real< T > tb = 1.0 / ta;

      CPPUNIT_ASSERT( b == tb );

      int ia( 2 );

      const Real< T > tbi = 1.0 / ( T ) ia;

      //cerr << tbi << std::endl;

      CPPUNIT_ASSERT( b == tbi );
   };
};

#endif /* RealTESTER_H_ */
