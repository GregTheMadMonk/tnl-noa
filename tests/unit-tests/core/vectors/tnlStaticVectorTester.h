/***************************************************************************
                          StaticVectorTester.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef StaticVectorTESTER_H_
#define StaticVectorTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Vectors/StaticVector.h>

using namespace TNL;

template< int Size, typename RealType >
class StaticVectorTester : public CppUnit :: TestCase
{
   public:

   typedef StaticVectorTester< Size, RealType > tnlStaticVectorTester;
   typedef CppUnit :: TestCaller< StaticVectorTester > TestCaller;

   StaticVectorTester(){};

   virtual
   ~StaticVectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "VectorTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testOperators", &StaticVectorTester::testOperators ) );
      return suiteOfTests;
   }

   void testOperators()
   {
      Vectors::StaticVector< Size, RealType > u1( 1.0 ), u2( 2.0 ), u3( 3.0 );

      u1 += u2;
      CPPUNIT_ASSERT( u1[ 0 ] == 3.0 );
      CPPUNIT_ASSERT( u1[ Size - 1 ] == 3.0 );

      u1 -= u2;
      CPPUNIT_ASSERT( u1[ 0 ] == 1.0 );
      CPPUNIT_ASSERT( u1[ Size - 1 ] == 1.0 );

      u1 *= 2.0;
      CPPUNIT_ASSERT( u1[ 0 ] == 2.0 );
      CPPUNIT_ASSERT( u1[ Size - 1 ] == 2.0 );

      u3 = u1 + u2;
      CPPUNIT_ASSERT( u3[ 0 ] == 4.0 );
      CPPUNIT_ASSERT( u3[ Size - 1 ] == 4.0 );

      u3 = u1 - u2;
      CPPUNIT_ASSERT( u3[ 0 ] == 0.0 );
      CPPUNIT_ASSERT( u3[ Size - 1 ] == 0.0 );

      u3 = u1 * 2.0;
      CPPUNIT_ASSERT( u3[ 0 ] == 4.0 );
      CPPUNIT_ASSERT( u3[ Size - 1 ] == 4.0 );

      CPPUNIT_ASSERT( u1 * u2 == 4.0 * Size );

      CPPUNIT_ASSERT( u1 < u3 );
      CPPUNIT_ASSERT( u1 <= u3 );
      CPPUNIT_ASSERT( u1 <= u2 );
      CPPUNIT_ASSERT( u3 > u1 );
      CPPUNIT_ASSERT( u3 >= u1 );
      CPPUNIT_ASSERT( u2 >= u1 );
   }

};



#else /* HAVE_CPPUNIT */
template< int Size, typename RealType >
class StaticVectorTester{};
#endif /* HAVE_CPPUNIT */




#endif /* StaticVectorTESTER_H_ */
