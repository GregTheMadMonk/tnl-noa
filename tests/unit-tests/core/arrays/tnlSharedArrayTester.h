/***************************************************************************
                          tnlSharedArrayTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSHAREDARRAYTESTER_H_
#define TNLSHAREDARRAYTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/arrays/tnlSharedArray.h>
#include <core/tnlFile.h>

using namespace TNL;

template< typename ElementType, typename Device, typename IndexType >
class tnlSharedArrayTester : public CppUnit :: TestCase
{
   public:
   tnlSharedArrayTester(){};

   virtual
   ~tnlSharedArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlSharedArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlSharedArrayTester< ElementType, Device, IndexType > >(
                               "testBind",
                               & tnlSharedArrayTester< ElementType, Device, IndexType > :: testBind )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlSharedArrayTester< ElementType, Device, IndexType > >(
                               "testComparisonOperator",
                               & tnlSharedArrayTester< ElementType, Device, IndexType > :: testComparisonOperator )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlSharedArrayTester< ElementType, Device, IndexType > >(
                               "testAssignmentOperator",
                               & tnlSharedArrayTester< ElementType, Device, IndexType > :: testAssignmentOperator )
                              );


      return suiteOfTests;
   }

   void testBind()
   {
      ElementType data[ 10 ];
      tnlSharedArray< ElementType, Device, IndexType > array;
      array. bind( data, 10 );
      for( int i = 0; i < 10; i ++ )
         data[ i ] = i;
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( data[ i ] == array. getElement( i ) );
   };

   void testComparisonOperator()
   {
      tnlSharedArray< ElementType, Device, IndexType > u, v, w;
      ElementType uData[ 10 ], vData[ 10 ], wData[ 10 ];
      u. bind( uData, 10 );
      v. bind( vData, 10 );
      w. bind( wData, 10 );
      for( int i = 0; i < 10; i ++ )
      {
         u. setElement( i, i );
         v. setElement( i, i );
         w. setElement( i, 2*i );
      }
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
      CPPUNIT_ASSERT( u != w );
      CPPUNIT_ASSERT( ! ( u == w ) );
   };

   void testAssignmentOperator()
   {
      tnlSharedArray< ElementType, Device, IndexType > u, v;
      ElementType uData[ 10 ], vData[ 10 ];
      u. bind( uData, 10 );
      v. bind( vData, 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
   };

   /*void testSave()
   {
      tnlSharedArray< ElementType, Device, IndexType > v( "test-array-v" );
      v. setSize( 100 );
      for( int i = 0; i < 100; i ++ )
         v. setElement( i, 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode );
      v. save( file );
      file. close();
      tnlSharedArray< ElementType, Device, IndexType > u( "test-array-u" );
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      tnlSharedArray< testingClassForArrayManagerTester >u ( "test-vector" );
   };*/

};


#endif /* TNLSHAREDARRAYTESTER_H_ */
