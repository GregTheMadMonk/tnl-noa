/***************************************************************************
                          VectorHostTester.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
#include <TNL/Containers/Vector.h>
#include <TNL/File.h>

using namespace TNL;

class testingClass
{

};

String getType( const testingClass& c )
{
   return String( "testingClass" );
};

template< class T > class VectorHostTester : public CppUnit :: TestCase
{
   public:
   VectorHostTester(){};

   virtual
   ~VectorHostTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "VectorHostTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testSharedData",
                               & VectorHostTester< T > :: testSharedData )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testSetGetElement",
                               & VectorHostTester< T > :: testSetGetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testComparisonOperator",
                               & VectorHostTester< T > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testEquivalenceOperator",
                               & VectorHostTester< T > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testSetValue",
                               & VectorHostTester< T > :: testSetValue )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testParallelReduciontMethods",
                               & VectorHostTester< T > :: testParallelReduciontMethods )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testBlasFunctions",
                               & VectorHostTester< T > :: testBlasFunctions )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testSaveAndLoad",
                               & VectorHostTester< T > :: testSaveAndLoad )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorHostTester< T > >(
                               "testUnusualStructures",
                               & VectorHostTester< T > :: testUnusualStructures )
                              );





      return suiteOfTests;
   }

   void testSharedData()
   {
      /*T data[ 10 ];
      Vector< T > u( "VectorTester :: u" );
      u. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = i;
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( data[ i ] == u. getElement( i ) );

      for( int i = 0; i < 10; i ++ )
         u. setElement( i, 2 * i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( data[ i ] == 2*i );

      u. setSize( 10 );
      u. setValue( 0 );

      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( data[ i ] == 2*i );*/

   };

   void testSetGetElement()
   {
      Vector< T > u( "VectorTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

   };

   void testComparisonOperator()
   {
      Vector< T > u( "VectorTester :: u" );
      Vector< T > v( "VectorTester :: v" );
      Vector< T > w( "VectorTester :: w" );
      u. setSize( 10 );
      v. setSize( 10 );
      w. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
      {
         u. setElement( i, i );
         v. setElement( i, i );
         w. setElement( i, 2*1 );
      }
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
      CPPUNIT_ASSERT( u != w );
      CPPUNIT_ASSERT( ! ( u == w ) );
   };

   void testEquivalenceOperator()
   {
      Vector< T > u( "VectorTester :: u" );
      Vector< T > v( "VectorTester :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
   };

   void testSetValue()
   {
      Vector< T > u( "VectorTester :: u" );
      u. setSize( 10 );
      for( int k = 0; k < 10; k ++ )
      {
         u. setValue( k );
         for( int i = 0; i < 10; i ++ )
            CPPUNIT_ASSERT( u. getElement( i ) == k );
      }
   };

   void testParallelReduciontMethods()
   {
      Vector< T > u( "VectorTester :: u" );
      u. setSize( 10 );

      for( int i = 0; i < 10; i ++ )
         u. setElement( i, -i );

      CPPUNIT_ASSERT( u. max() == 0 );
      CPPUNIT_ASSERT( u. min() == - 9 );
      CPPUNIT_ASSERT( u. absMax() == 9 );
      CPPUNIT_ASSERT( u. absMin() == 0 );
      CPPUNIT_ASSERT( u. lpNorm( ( T ) 1 ) == 45 );
      CPPUNIT_ASSERT( u. sum() == -45 );
   };

   void testBlasFunctions()
   {
      Vector< T > u( "VectorTester :: u" );
      Vector< T > v( "VectorTester :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      u. setValue( 2 );
      v. setValue( 3 );

      CPPUNIT_ASSERT( u. scalarProduct( v ) == 60 );

      u. alphaXPlusY( ( T ) 2, v );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == 8 );
   }

   void testSaveAndLoad()
   {
      Vector< T, Devices::Host > v( "test-long-vector-u" );
      v. setSize( 100 );
      v. setValue( 3.14147 );
      File file;
      file. open( "test-file.tnl", tnlWriteMode );
      v. save( file );
      file. close();
      Vector< T, Devices::Host > u( "test-long-vector-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      Vector< testingClass >u ( "test-vector" );
   };

};


#endif /* TNLLONGVECTORHOSTTESTER_H_ */
