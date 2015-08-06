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

#ifndef TNLARRAYTESTER_H_
#define TNLARRAYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/arrays/tnlArray.h>
#include <core/tnlFile.h>

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( tnlArray< ElementType, tnlCuda, IndexType >* u );
#endif

class testingClassForArrayTester
{
   public:

      static tnlString getType()
      {
         return tnlString( "testingClassForArrayTester" );
      };
};

tnlString getType( const testingClassForArrayTester& c )
{
   return tnlString( "testingClassForArrayTester" );
};

template< typename ElementType, typename Device, typename IndexType >
class tnlArrayTester : public CppUnit :: TestCase
{
   public:

   typedef tnlArrayTester< ElementType, Device, IndexType > ArrayTester;
   typedef CppUnit :: TestCaller< ArrayTester > TestCaller;

   tnlArrayTester(){};

   virtual
   ~tnlArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testConstructorDestructor", &ArrayTester::testConstructorDestructor ) );
      suiteOfTests -> addTest( new TestCaller( "testSetSize", &ArrayTester::testSetSize ) );
      suiteOfTests -> addTest( new TestCaller( "testSetGetElement", &ArrayTester::testSetGetElement ) );
      suiteOfTests -> addTest( new TestCaller( "testComparisonOperator", &ArrayTester::testComparisonOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testAssignmentOperator", &ArrayTester::testAssignmentOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testGetSize", &ArrayTester::testGetSize ) );
      suiteOfTests -> addTest( new TestCaller( "testReset", &ArrayTester::testReset ) );
      suiteOfTests -> addTest( new TestCaller( "testSetSizeAndDestructor", &ArrayTester::testSetSizeAndDestructor ) );
      suiteOfTests -> addTest( new TestCaller( "testSaveAndLoad", &ArrayTester::testSaveAndLoad ) );
      suiteOfTests -> addTest( new TestCaller( "testUnusualStructures",  &ArrayTester::testUnusualStructures ) );
      return suiteOfTests;
   }

   void testConstructorDestructor()
   {
      tnlArray< ElementType, Device, IndexType > u;
   }

   void testSetSize()
   {
      tnlArray< ElementType, Device, IndexType > u, v;
      u. setSize( 10 );
      v. setSize( 10 );
   }

   void testSetGetElement()
   {
      tnlArray< ElementType, Device, IndexType > u( "tnlArrayTester :: u" );
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

      u.setValue( 0 );
      if( Device::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 10; i ++ )
            u[ i ] =  i;
      }
      if( Device::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         tnlArray< ElementType, Device, IndexType >* kernel_u =
                  tnlCuda::passToDevice( u );
         testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
         tnlCuda::freeFromDevice( kernel_u );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( u.getElement( i ) == i );
   };

   void testComparisonOperator()
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
   };

   void testAssignmentOperator()
   {
      tnlArray< ElementType, Device, IndexType > u;
      tnlArray< ElementType, Device, IndexType > v;
      u. setName( "tnlArrayTester :: testAssignmentOperator :: u" );
      v. setName( "tnlArrayTester :: testAssignmentOperator :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( v == u );
      CPPUNIT_ASSERT( ! ( u != v ) );
      CPPUNIT_ASSERT( ! ( v != u ) );

      v.setValue( 0 );
      tnlArray< ElementType, tnlHost, IndexType > w;
      w.setSize( 10 );
      w = u;

      CPPUNIT_ASSERT( u == w );
      CPPUNIT_ASSERT( ! ( u != w ) );

      v.setValue( 0 );
      v = w;
      CPPUNIT_ASSERT( v == w );
      CPPUNIT_ASSERT( ! ( v != w ) );
   };

   void testGetSize()
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
      file. open( "test-file.tnl", tnlWriteMode );
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
      tnlArray< testingClassForArrayTester >u ( "test-array" );
   };

};

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( tnlArray< ElementType, tnlCuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getSize() )
      ( *u )[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

#endif /* HAVE_CPPUNIT */

#endif /* TNLARRAYTESTER_H_ */
