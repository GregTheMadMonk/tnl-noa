/***************************************************************************
                          tnlArrayTester.h -  description
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
#include <core/tnlArray.h>
#include <core/tnlFile.h>


class testingClassForArrayManagerTester
{

};

tnlString GetParameterType( const testingClassForArrayManagerTester& c )
{
   return tnlString( "testingClassForArrayManagerTester" );
};

template< typename ElementType, typename Device, typename IndexType >
class tnlArrayTester : public CppUnit :: TestCase
{
   public:
   tnlArrayTester(){};

   virtual
   ~tnlArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSharedData",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSharedData )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSetGetElement",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSetGetElement )
                              );
      /*suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testComparisonOperator",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testComparisonOperator )
                              );*/
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testEquivalenceOperator",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSetSize",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSetSize )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testReset",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testReset )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSetSizeAndDestructor",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSetSizeAndDestructor )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSaveAndLoad",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSaveAndLoad )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testUnusualStructures",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testUnusualStructures )
                              );
      return suiteOfTests;
   }

   void testSharedData()
   {
      ElementType data[ 10 ];
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: u" );
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
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );
   };

   /*void testComparisonOperator()
   {
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: u" );
      tnlArray< ElementType, Device, IndexType > v( "tnlArrayTester :: v" );
      tnlArray< ElementType, Device, IndexType > w( "tnlArrayTester :: w" );
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
   };*/

   void testEquivalenceOperator()
   {
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: u" );
      tnlArray< ElementType, Device, IndexType > v( "tnlArrayTester :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
   };

   void testSetSize()
   {
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: testSetSize - u" );
      const int maxSize = 10;
      for( int i = 0; i < maxSize; i ++ )
         u. setSize( i );

      CPPUNIT_ASSERT( u. getSize() == maxSize - 1 );
   };

   void testReset()
   {
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: testReset - u" );
      u. setSize( 100 );
      CPPUNIT_ASSERT( u. getSize() == 100 );
      u. reset();
      CPPUNIT_ASSERT( u. getSize() == 0 );
      u. setSize( 100 );
      CPPUNIT_ASSERT( u. getSize() == 100 );
      u. reset();
      CPPUNIT_ASSERT( u. getSize() == 0 );

   };

   void testSetSizeAndDestructor()
   {
      for( int i = 0; i < 100; i ++ )
      {
         tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: testSetSizeAndDestructor - u" );
         u. setSize( i );
      }
   }

   void testSaveAndLoad()
   {
      tnlArray< ElementType, Device, IndexType > v( "test-array-v" );
      v. setSize( 100 );
      for( int i = 0; i < 100; i ++ )
         v. setElement( i, 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode, tnlCompressionBzip2 );
      v. save( file );
      file. close();
      tnlArray< ElementType, Device, IndexType > u( "test-array-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      tnlArray< testingClassForArrayManagerTester >u ( "test-vector" );
   };

};


#endif /* TNLARRAYMANAGERTESTER_H_ */
