/***************************************************************************
                          VectorTester.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef VectorHOSTTESTER_H_
#define VectorHOSTTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/File.h>
#include <TNL/core/mfuncs.h>

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class VectorTester : public CppUnit :: TestCase
{
   public:

   typedef Vectors::Vector< RealType, Device, IndexType > VectorType;
   typedef VectorTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   VectorTester(){};

   virtual
   ~VectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "VectorTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "testMax", &TesterType::testMax ) );
      suiteOfTests -> addTest( new TestCallerType( "testMin", &TesterType::testMin ) );
      suiteOfTests -> addTest( new TestCallerType( "testAbsMax", &TesterType::testAbsMax ) );
      suiteOfTests -> addTest( new TestCallerType( "testAbsMin", &TesterType::testAbsMin ) );
      suiteOfTests -> addTest( new TestCallerType( "testLpNorm", &TesterType::testLpNorm ) );
      suiteOfTests -> addTest( new TestCallerType( "testSum", &TesterType::testSum ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceMax", &TesterType::testDifferenceMax ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceMin", &TesterType::testDifferenceMin ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceAbsMax", &TesterType::testDifferenceAbsMax ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceAbsMin", &TesterType::testDifferenceAbsMin ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceLpNorm", &TesterType::testDifferenceLpNorm ) );
      suiteOfTests -> addTest( new TestCallerType( "testDifferenceSum", &TesterType::testDifferenceSum ) );
      suiteOfTests -> addTest( new TestCallerType( "testScalarMultiplication", &TesterType::testScalarMultiplication ) );
      suiteOfTests -> addTest( new TestCallerType( "testScalarProduct", &TesterType::testScalarProduct ) );
      suiteOfTests -> addTest( new TestCallerType( "addVectorTest", &TesterType::addVectorTest ) );
      return suiteOfTests;
   }

   void testMax()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v. setElement( i, i );
      CPPUNIT_ASSERT( v. max() == 9 );
   };

   void testMin()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v. setElement( i, i );
      CPPUNIT_ASSERT( v. min() == 0 );
   };

   void testAbsMax()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v.setElement( i, -i );
      CPPUNIT_ASSERT( v. absMax() == 9 );
   };

   void testAbsMin()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v.setElement( i,  -i );
      CPPUNIT_ASSERT( v. absMin() == 0 );
   };

   void testLpNorm()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v.setElement(  i, -2 );
      CPPUNIT_ASSERT( isSmall( v.lpNorm( 1 ) - 20.0 ) );
      CPPUNIT_ASSERT( isSmall( v.lpNorm( 2 ) - ::sqrt( 40.0 ) ) );
      CPPUNIT_ASSERT( isSmall( v.lpNorm( 3 ) - ::pow( 80.0, 1.0/3.0 ) ) );
   };

   void testSum()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v.setElement( i, -2 );
      CPPUNIT_ASSERT( v. sum() == -20.0 );
      for( int i = 0; i < 10; i ++ )
         v.setElement( i,  2 );
      CPPUNIT_ASSERT( v. sum() == 20.0 );

   };

   void testDifferenceMax()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i,  i );
         v2.setElement( i, -i );
      }
      CPPUNIT_ASSERT( v1. differenceMax( v2 ) == 18.0 );
   };

   void testDifferenceMin()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, i );
         v2.setElement( i, -i );
      }
      CPPUNIT_ASSERT( v1. differenceMin( v2 ) == 0.0 );
   };

   void testDifferenceAbsMax()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, -i );
         v2.setElement( i, i );
      }
      CPPUNIT_ASSERT( v1. differenceAbsMax( v2 ) == 18.0 );
   };

   void testDifferenceAbsMin()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, -i );
         v2.setElement( i, i );
      }
      CPPUNIT_ASSERT( v1. differenceAbsMin( v2 ) == 0.0 );
   };

   void testDifferenceLpNorm()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, -1 );
         v2.setElement( i, 1 );
      }
      CPPUNIT_ASSERT( isSmall( v1.differenceLpNorm( v2, 1.0 ) - 20.0 ) );
      CPPUNIT_ASSERT( isSmall( v1.differenceLpNorm( v2, 2.0 ) - ::sqrt( 40.0 ) ) );
      CPPUNIT_ASSERT( isSmall( v1.differenceLpNorm( v2, 3.0 ) - ::pow( 80.0, 1.0/3.0 ) ) );
   };

   void testDifferenceSum()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, -1 );
         v2.setElement( i, 1 );
      }
      CPPUNIT_ASSERT( v1. differenceSum( v2 ) == -20.0 );
   };

   void testScalarMultiplication()
   {
      Vectors::Vector< RealType, Device, IndexType > v;
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         v.setElement( i, i );
      v. scalarMultiplication( 5.0 );

      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v. getElement( i ) == 5 * i );
   };

   void testScalarProduct()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      v1.setElement( 0, -1 );
      v2.setElement( 0, 1 );
      for( int i = 1; i < 10; i ++ )
      {
         v1.setElement( i, v1.getElement( i - 1 ) * -1 );
         v2.setElement( i, v2.getElement( i - 1 ) );
      }
      CPPUNIT_ASSERT( v1. scalarProduct( v2 ) == 0.0 );
   };

   void addVectorTest()
   {
      Vectors::Vector< RealType, Device, IndexType > v1, v2;
      v1. setSize( 10 );
      v2. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         v1.setElement( i, i );
         v2.setElement( i, 2.0 * i );
      }
      v1. addVector( v2, 2.0 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( v1. getElement( i ) == 5.0 * i );
   };

};

#else /* HAVE_CPPUNIT */

template< typename RealType, typename Device, typename IndexType >
class VectorTester{};

#endif /* HAVE_CPPUNIT */

#endif /* VectorHOSTTESTER_H_ */
