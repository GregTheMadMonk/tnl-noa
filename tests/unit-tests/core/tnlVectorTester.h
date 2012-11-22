/***************************************************************************
                          tnlVectorTester.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORHOSTTESTER_H_
#define TNLLONGVECTORHOSTTESTER_H_
/*
 *
 */
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlVector.h>
#include <core/tnlFile.h>

template< typename RealType, typename Device, typename IndexType >
class tnlVectorTester : public CppUnit :: TestCase
{
   public:
   tnlVectorTester(){};

   virtual
   ~tnlVectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlVectorTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testMax",
                                & tnlVectorTester< RealType, Device, IndexType > :: testMax )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testMin",
                                & tnlVectorTester< RealType, Device, IndexType > :: testMin )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testAbsMax",
                                & tnlVectorTester< RealType, Device, IndexType > :: testAbsMax )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testAbsMin",
                                & tnlVectorTester< RealType, Device, IndexType > :: testAbsMin )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testLpNorm",
                                & tnlVectorTester< RealType, Device, IndexType > :: testLpNorm )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testSum",
                                & tnlVectorTester< RealType, Device, IndexType > :: testSum )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceMax",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceMax )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceMin",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceMin )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceAbsMax",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceAbsMax )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceAbsMin",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceAbsMin )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceLpNorm",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceLpNorm )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testDifferenceSum",
                                & tnlVectorTester< RealType, Device, IndexType > :: testDifferenceSum )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testScalarMultiplication",
                                & tnlVectorTester< RealType, Device, IndexType > :: testScalarMultiplication )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testSdot",
                                & tnlVectorTester< RealType, Device, IndexType > :: testSdot )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testSaxpy",
                                & tnlVectorTester< RealType, Device, IndexType > :: testSaxpy )
                                );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorTester< RealType, Device, IndexType > >(
                                "testSaxpy",
                                & tnlVectorTester< RealType, Device, IndexType > :: testSaxpy )
                                );
      return suiteOfTests;
   }

   void testMax()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v. setElement( i, i );
      CPPUNIT_ASSERT( v. max() == 9 );
   };

   void testMin()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = i;
      CPPUNIT_ASSERT( v. min() == 0 );
   };

   void testAbsMax()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = -i;
      CPPUNIT_ASSERT( v. absMax() == 9 );
   };

   void testAbsMin()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = -i;
      CPPUNIT_ASSERT( v. absMin() == 0 );
   };

   void testLpNorm()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = -2;
      CPPUNIT_ASSERT( v. lpNorm( 1 ) == 20.0 );
      CPPUNIT_ASSERT( v. lpNorm( 2 ) == sqrt( 40.0 ) );
      CPPUNIT_ASSERT( v. lpNorm( 3 ) == pow( 80.0, 1.0/3.0 ) );
   };

   void testSum()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = -2;
      CPPUNIT_ASSERT( v. sum() == -20.0 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = 2;
      CPPUNIT_ASSERT( v. sum() == 20.0 );

   };

   void testDifferenceMax()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = i;
         v2[ i ] = -i;
      }
      CPPUNIT_ASSERT( v1. differenceMax( v2 ) == 18.0 );
   };

   void testDifferenceMin()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = i;
         v2[ i ] = -i;
      }
      CPPUNIT_ASSERT( v1. differenceMin( v2 ) == 0.0 );
   };

   void testDifferenceAbsMax()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = -i;
         v2[ i ] = i;
      }
      CPPUNIT_ASSERT( v1. differenceAbsMax( v2 ) == 18.0 );
   };

   void testDifferenceAbsMin()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = -i;
         v2[ i ] = i;
      }
      CPPUNIT_ASSERT( v1. differenceAbsMin( v2 ) == 0.0 );
   };

   void testDifferenceLpNorm()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = -1;
         v2[ i ] = 1;
      }
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 1.0 ) == 20.0 );
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 2.0 ) == sqrt( 40.0 ) );
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 3.0 ) == pow( 80.0, 1.0/3.0 ) );
   };

   void testDifferenceSum()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = -1;
         v2[ i ] = 1;
      }
      CPPUNIT_ASSERT( v1. differenceSum( v2 ) == -20.0 );
   };

   void testScalarMultiplication()
   {
      tnlVector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v[ i ] = i;
      v. scalarMultiplication( 5.0 );

      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v. getElement( i ) == 5 * i );
   };

   void testSdot()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      v1[ 0 ] = -1;
      v2[ 0 ] = 1;
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = v1[ i - 1 ] * -1;
         v2[ i ] = v2[ i - 1 ];
      }
      CPPUNIT_ASSERT( v1. sdot( v2 ) == 0.0 );
   };

   void testSaxpy()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = i;
         v2[ i ] = 2.0 * i;
      }
      v1. saxpy( 2.0, v2 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v1. getElement( i ) == 5.0 * i );
   };

   void testSaxmy()
   {
      tnlVector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1[ i ] = i;
         v2[ i ] = 2.0 * i;
      }
      v1. saxmy( 2.0, v2 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v1. getElement( i ) == 3.0 * i );
   };
};


#endif /* TNLLONGVECTORHOSTTESTER_H_ */
