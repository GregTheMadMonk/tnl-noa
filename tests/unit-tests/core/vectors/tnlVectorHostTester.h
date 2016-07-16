/***************************************************************************
                          tnlVectorHostTester.h  -  description
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
#include <core/vectors/tnlVector.h>
#include <core/tnlFile.h>


class testingClass
{

};

tnlString getType( const testingClass& c )
{
   return tnlString( "testingClass" );
};

template< class T > class tnlVectorHostTester : public CppUnit :: TestCase
{
   public:
   tnlVectorHostTester(){};

   virtual
   ~tnlVectorHostTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlVectorHostTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testSharedData",
                               & tnlVectorHostTester< T > :: testSharedData )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testSetGetElement",
                               & tnlVectorHostTester< T > :: testSetGetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testComparisonOperator",
                               & tnlVectorHostTester< T > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testEquivalenceOperator",
                               & tnlVectorHostTester< T > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testSetValue",
                               & tnlVectorHostTester< T > :: testSetValue )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testParallelReduciontMethods",
                               & tnlVectorHostTester< T > :: testParallelReduciontMethods )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testBlasFunctions",
                               & tnlVectorHostTester< T > :: testBlasFunctions )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testSaveAndLoad",
                               & tnlVectorHostTester< T > :: testSaveAndLoad )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorHostTester< T > >(
                               "testUnusualStructures",
                               & tnlVectorHostTester< T > :: testUnusualStructures )
                              );





      return suiteOfTests;
   }

   void testSharedData()
   {
      /*T data[ 10 ];
      tnlVector< T > u( "tnlVectorTester :: u" );
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
      tnlVector< T > u( "tnlVectorTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

   };

   void testComparisonOperator()
   {
      tnlVector< T > u( "tnlVectorTester :: u" );
      tnlVector< T > v( "tnlVectorTester :: v" );
      tnlVector< T > w( "tnlVectorTester :: w" );
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
      tnlVector< T > u( "tnlVectorTester :: u" );
      tnlVector< T > v( "tnlVectorTester :: v" );
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
      tnlVector< T > u( "tnlVectorTester :: u" );
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
      tnlVector< T > u( "tnlVectorTester :: u" );
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
      tnlVector< T > u( "tnlVectorTester :: u" );
      tnlVector< T > v( "tnlVectorTester :: v" );
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
      tnlVector< T, tnlHost > v( "test-long-vector-u" );
      v. setSize( 100 );
      v. setValue( 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode );
      v. save( file );
      file. close();
      tnlVector< T, tnlHost > u( "test-long-vector-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      tnlVector< testingClass >u ( "test-vector" );
   };

};


#endif /* TNLLONGVECTORHOSTTESTER_H_ */
