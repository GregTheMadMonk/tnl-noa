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

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/arrays/tnlArray.h>
#include <core/tnlFile.h>


class testingClassForArrayManagerTester
{
   public:

      static tnlString getType()
      {
         return tnlString( "testingClassForArrayManagerTester" );
      };
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
                               "testConstructorDestructor",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testConstructorDestructor )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSetSize",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSetSize )
                              );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testSetGetElement",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testSetGetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testComparisonOperator",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testEquivalenceOperator",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testEquivalenceOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< ElementType, Device, IndexType > >(
                               "testGetSize",
                               & tnlArrayTester< ElementType, Device, IndexType > :: testGetSize )
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

   void testEquivalenceOperator()
   {
      tnlArray< ElementType, Device, IndexType > u;
      tnlArray< ElementType, Device, IndexType > v;
      u. setName( "tnlArrayTester :: testEquivalenceOperator :: u" );
      v. setName( "tnlArrayTester :: testEquivalenceOperator :: v" );
      u. setSize( 10 );
      v. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      //CPPUNIT_ASSERT( u == v );
      //CPPUNIT_ASSERT( ! ( u != v ) );
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
      tnlArray< testingClassForArrayManagerTester >u ( "test-vector" );
   };

};
#else /* HAVE_CPPUNIT */
template< typename ElementType, typename Device, typename IndexType >
class tnlArrayTester{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLARRAYMANAGERTESTER_H_ */
