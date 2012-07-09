/***************************************************************************
                          tnlArrayManagerTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLARRAYMANAGERTESTER_H_
#define TNLARRAYMANAGERTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlArrayManager.h>
#include <core/tnlFile.h>


class testingClassForArrayManagerTester
{

};

tnlString GetParameterType( const testingClassForArrayManagerTester& c )
{
   return tnlString( "testingClassForArrayManagerTester" );
};

template< typename ElementType, tnlDevice Device, typename IndexType >
class tnlArrayManagerTester : public CppUnit :: TestCase
{
   public:
   tnlArrayManagerTester(){};

   virtual
   ~tnlArrayManagerTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayManagerTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testSharedData",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testSharedData )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testSetGetElement",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testSetGetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testComparisonOperator",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testEquivalenceOperator",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testSaveAndLoad",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testSaveAndLoad )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayManagerTester< ElementType, Device, IndexType > >(
                               "testUnusualStructures",
                               & tnlArrayManagerTester< ElementType, Device, IndexType > :: testUnusualStructures )
                              );





      return suiteOfTests;
   }

   void testSharedData()
   {
      ElementType data[ 10 ];
      tnlArrayManager< ElementType, Device, IndexType > u( "tnlArrayManagerTester :: u" );
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
         for( int i = 0; i < 10; i ++ )
            u. setElement( i, 0 );

      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( data[ i ] == 2*i );

   };

   void testSetGetElement()
   {
      tnlArrayManager< ElementType, Device, IndexType > u( "tnlArrayManagerTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

   };

   void testComparisonOperator()
   {
      tnlArrayManager< ElementType, Device, IndexType > u( "tnlArrayManagerTester :: u" );
      tnlArrayManager< ElementType, Device, IndexType > v( "tnlArrayManagerTester :: v" );
      tnlArrayManager< ElementType, Device, IndexType > w( "tnlArrayManagerTester :: w" );
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
      tnlArrayManager< ElementType, Device, IndexType > u( "tnlArrayManagerTester :: u" );
      tnlArrayManager< ElementType, Device, IndexType > v( "tnlArrayManagerTester :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
   };

   void testSaveAndLoad()
   {
      tnlArrayManager< ElementType, Device, IndexType > v( "test-array-v" );
      v. setSize( 100 );
      for( int i = 0; i < 100; i ++ )
         v. setElement( i, 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode, tnlCompressionBzip2 );
      v. save( file );
      file. close();
      tnlArrayManager< ElementType, Device, IndexType > u( "test-array-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      tnlArrayManager< testingClassForArrayManagerTester >u ( "test-vector" );
   };

};


#endif /* TNLARRAYMANAGERTESTER_H_ */
