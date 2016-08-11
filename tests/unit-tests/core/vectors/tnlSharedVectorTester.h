/***************************************************************************
                          SharedVectorTester.h -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef SharedVectorTESTER_H_
#define SharedVectorTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/File.h>

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class SharedVectorTester : public CppUnit :: TestCase
{
   public:
   SharedVectorTester(){};

   virtual
   ~SharedVectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "SharedVectorTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testMax",
                               & SharedVectorTester< RealType, Device, IndexType > :: testMax )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testMin",
                               & SharedVectorTester< RealType, Device, IndexType > :: testMin )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testAbsMax",
                               & SharedVectorTester< RealType, Device, IndexType > :: testAbsMax )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testAbsMin",
                               & SharedVectorTester< RealType, Device, IndexType > :: testAbsMin )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testLpNorm",
                               & SharedVectorTester< RealType, Device, IndexType > :: testLpNorm )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testSum",
                               & SharedVectorTester< RealType, Device, IndexType > :: testSum )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceMax",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceMax )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceMin",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceMin )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceAbsMax",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceAbsMax )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceAbsMin",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceAbsMin )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceLpNorm",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceLpNorm )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testDifferenceSum",
                               & SharedVectorTester< RealType, Device, IndexType > :: testDifferenceSum )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testScalarMultiplication",
                               & SharedVectorTester< RealType, Device, IndexType > :: testScalarMultiplication )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testScalarProduct",
                               & SharedVectorTester< RealType, Device, IndexType > :: testScalarProduct )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testalphaXPlusY",
                               & SharedVectorTester< RealType, Device, IndexType > :: testalphaXPlusY )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SharedVectorTester< RealType, Device, IndexType > >(
                               "testalphaXPlusY",
                               & SharedVectorTester< RealType, Device, IndexType > :: testalphaXPlusY )
                               );
      return suiteOfTests;
   }

   void testMax()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = i;
      CPPUNIT_ASSERT( v. max() == 9 );
   };

   void testMin()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = i;
      CPPUNIT_ASSERT( v. min() == 0 );
   };

   void testAbsMax()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = -i;
      CPPUNIT_ASSERT( v. absMax() == 9 );
   };

   void testAbsMin()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = -i;
      CPPUNIT_ASSERT( v. absMin() == 0 );
   };

   void testLpNorm()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = -2;
      CPPUNIT_ASSERT( v. lpNorm( 1 ) == 20.0 );
      CPPUNIT_ASSERT( v. lpNorm( 2 ) == ::sqrt( 40.0 ) );
      CPPUNIT_ASSERT( v. lpNorm( 3 ) == ::pow( 80.0, 1.0/3.0 ) );
   };

   void testSum()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = -2;
      CPPUNIT_ASSERT( v. sum() == -20.0 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = 2;
      CPPUNIT_ASSERT( v. sum() == 20.0 );

   };

   void testDifferenceMax()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = i;
         data2[ i ] = -i;
      }
      CPPUNIT_ASSERT( v1. differenceMax( v2 ) == 18.0 );
   };

   void testDifferenceMin()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = i;
         data2[ i ] = -i;
      }
      CPPUNIT_ASSERT( v1. differenceMin( v2 ) == 0.0 );
   };

   void testDifferenceAbsMax()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = -i;
         data2[ i ] = i;
      }
      CPPUNIT_ASSERT( v1. differenceAbsMax( v2 ) == 18.0 );
   };

   void testDifferenceAbsMin()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = -i;
         data2[ i ] = i;
      }
      CPPUNIT_ASSERT( v1. differenceAbsMin( v2 ) == 0.0 );
   };

   void testDifferenceLpNorm()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = -1;
         data2[ i ] = 1;
      }
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 1.0 ) == 20.0 );
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 2.0 ) == ::sqrt( 40.0 ) );
      CPPUNIT_ASSERT( v1. differenceLpNorm( v2, 3.0 ) == ::pow( 80.0, 1.0/3.0 ) );
   };

   void testDifferenceSum()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = -1;
         data2[ i ] = 1;
      }
      CPPUNIT_ASSERT( v1. differenceSum( v2 ) == -20.0 );
   };

   void testScalarMultiplication()
   {
      RealType data[ 10 ];
      SharedVector< RealType, Device, IndexType > v;
      v. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = i;
      v. scalarMultiplication( 5.0 );

      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v. getElement( i ) == 5 * i );
   };

   void testScalarProduct()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      data1[ 0 ] = -1;
      data2[ 0 ] = 1;
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = data1[ i - 1 ] * -1;
         data2[ i ] = data2[ i - 1 ];
      }
      CPPUNIT_ASSERT( v1. scalarProduct( v2 ) == 0.0 );
   };

   void testalphaXPlusY()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = i;
         data2[ i ] = 2.0 * i;
      }
      v1. alphaXPlusY( 2.0, v2 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v1. getElement( i ) == 5.0 * i );
   };

   void testSaxmy()
   {
      RealType data1[ 10 ], data2[ 10 ];
      SharedVector< RealType, Device, IndexType > v1, v2;
      v1. bind( data1, 10 );
      v2. bind( data2, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         data1[ i ] = i;
         data2[ i ] = 2.0 * i;
      }
      v1. saxmy( 2.0, v2 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v1. getElement( i ) == 3.0 * i );
   };
};


#endif /* SharedArrayTESTER_H_ */
