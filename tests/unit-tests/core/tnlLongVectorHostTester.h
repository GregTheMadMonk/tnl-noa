/***************************************************************************
                          tnlLongVectorHostTester.h  -  description
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
#include <core/tnlLongVectorHost.h>
#include <core/tnlFile.h>


class testingClass
{

};

tnlString GetParameterType( const testingClass& c )
{
   return tnlString( "testingClass" );
};

template< class T > class tnlLongVectorHostTester : public CppUnit :: TestCase
{
   public:
   tnlLongVectorHostTester(){};

   virtual
   ~tnlLongVectorHostTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlLongVectorHostTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testSharedData",
                               & tnlLongVectorHostTester< T > :: testSharedData )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testSetGetElement",
                               & tnlLongVectorHostTester< T > :: testSetGetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testComparisonOperator",
                               & tnlLongVectorHostTester< T > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testEquivalenceOperator",
                               & tnlLongVectorHostTester< T > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testSetValue",
                               & tnlLongVectorHostTester< T > :: testSetValue )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testParallelReduciontMethods",
                               & tnlLongVectorHostTester< T > :: testParallelReduciontMethods )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testBlasFunctions",
                               & tnlLongVectorHostTester< T > :: testBlasFunctions )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testSaveAndLoad",
                               & tnlLongVectorHostTester< T > :: testSaveAndLoad )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorHostTester< T > >(
                               "testUnusualStructures",
                               & tnlLongVectorHostTester< T > :: testUnusualStructures )
                              );





      return suiteOfTests;
   }

   void testSharedData()
   {
      T data[ 10 ];
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      u. setSharedData( data, 10 );
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
         CPPUNIT_ASSERT( data[ i ] == 2*i );

   };

   void testSetGetElement()
   {
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

   };

   void testComparisonOperator()
   {
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      tnlLongVector< T > v( "tnlLongVectorTester :: v" );
      tnlLongVector< T > w( "tnlLongVectorTester :: w" );
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
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      tnlLongVector< T > v( "tnlLongVectorTester :: v" );
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
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
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
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      u. setSize( 10 );

      for( int i = 0; i < 10; i ++ )
         u. setElement( i, -i );

      CPPUNIT_ASSERT( tnlMax( u ) == 0 );
      CPPUNIT_ASSERT( tnlMin( u ) == - 9 );
      CPPUNIT_ASSERT( tnlAbsMax( u ) == 9 );
      CPPUNIT_ASSERT( tnlAbsMin( u ) == 0 );
      CPPUNIT_ASSERT( tnlLpNorm( u, ( T ) 1 ) == 45 );
      CPPUNIT_ASSERT( tnlSum( u ) == -45 );
   };

   void testBlasFunctions()
   {
      tnlLongVector< T > u( "tnlLongVectorTester :: u" );
      tnlLongVector< T > v( "tnlLongVectorTester :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      u. setValue( 2 );
      v. setValue( 3 );

      CPPUNIT_ASSERT( tnlSDOT( u, v ) == 60 );

      tnlSAXPY( ( T ) 2, v, u );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == 8 );
   }

   void testSaveAndLoad()
   {
      tnlLongVector< T, tnlHost > v( "test-long-vector-u" );
      v. setSize( 100 );
      v. setValue( 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode, tnlCompressionBzip2 );
      v. save( file );
      file. close();
      tnlLongVector< T, tnlHost > u( "test-long-vector-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      tnlLongVector< testingClass >u ( "test-vector" );
   };

};


#endif /* TNLLONGVECTORHOSTTESTER_H_ */
