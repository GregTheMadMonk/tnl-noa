/***************************************************************************
                          tnlStaticVectorTester.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSTATICVECTORTESTER_H_
#define TNLSTATICVECTORTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/vectors/tnlStaticVector.h>


template< int Size, typename RealType >
class tnlStaticVectorTester : public CppUnit :: TestCase
{
   public:

   typedef tnlStaticVectorTester< Size, RealType > StaticVectorTester;
   typedef CppUnit :: TestCaller< StaticVectorTester > TestCaller;

   tnlStaticVectorTester(){};

   virtual
   ~tnlStaticVectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlVectorTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testOperators", &StaticVectorTester::testOperators ) );
      return suiteOfTests;
   }

   void testOperators()
   {
      tnlStaticVector< Size, RealType > u1( 1.0 ), u2( 2.0 ), u3( 3.0 );

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
class tnlStaticVectorTester{};
#endif /* HAVE_CPPUNIT */




#endif /* TNLSTATICVECTORTESTER_H_ */
