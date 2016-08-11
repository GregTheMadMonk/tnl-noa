/***************************************************************************
                          ArrayTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLARRAYTESTER_H_
#define TNLARRAYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Containers/Array.h>
#include <TNL/File.h>

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ElementType, Devices::Cuda, IndexType >* u );
#endif

class testingClassForArrayTester
{
   public:

      static String getType()
      {
         return String( "testingClassForArrayTester" );
      };
};

String getType( const testingClassForArrayTester& c )
{
   return String( "testingClassForArrayTester" );
};

template< typename ElementType, typename Device, typename IndexType >
class ArrayTester : public CppUnit :: TestCase
{
   public:

   typedef ArrayTester< ElementType, Device, IndexType > ArrayTesterType;
   typedef CppUnit :: TestCaller< ArrayTesterType > TestCaller;
   typedef Array< ElementType, Device, IndexType > ArrayType;

   ArrayTester(){};

   virtual
   ~ArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "ArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testConstructorDestructor", &ArrayTester::testConstructorDestructor ) );
      suiteOfTests -> addTest( new TestCaller( "testSetSize", &ArrayTester::testSetSize ) );
      suiteOfTests -> addTest( new TestCaller( "testBind", &ArrayTester::testBind ) );
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
      ArrayType u;
      ArrayType v( 10 );
      CPPUNIT_ASSERT( v.getSize() == 10 );
   }

   void testSetSize()
   {
      ArrayType u, v;
      u.setSize( 10 );
      v.setSize( 10 );
      CPPUNIT_ASSERT( u.getSize() == 10 );
      CPPUNIT_ASSERT( v.getSize() == 10 );
   }
 
   void testBind()
   {
      ArrayType u( 10 ), v;
      u.setValue( 27 );
      v.bind( u );
      CPPUNIT_ASSERT( v.getSize() == u.getSize() );
      CPPUNIT_ASSERT( u.getElement( 0 ) == 27 );
      v.setValue( 50 );
      CPPUNIT_ASSERT( u.getElement( 0 ) == 50 );
      u.reset();
      CPPUNIT_ASSERT( u.getSize() == 0 );
      CPPUNIT_ASSERT( v.getElement( 0 ) == 50 );
 
      ElementType data[ 10 ] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
      u.bind( data, 10 );
      CPPUNIT_ASSERT( u.getElement( 1 ) == 2 );
      v.bind( u );
      CPPUNIT_ASSERT( v.getElement( 1 ) == 2 );
      u.reset();
      v.setElement( 1, 3 );
      v.reset();
      CPPUNIT_ASSERT( data[ 1 ] == 3 );
   }

   void testSetGetElement()
   {
      using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > u;
      u. setSize( 10 );
      for( int i = 0; i < 10; i ++ )
         u. setElement( i, i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u. getElement( i ) == i );

      u.setValue( 0 );
      if( std::is_same< Device, Devices::Host >::value )
      {
         for( int i = 0; i < 10; i ++ )
            u[ i ] =  i;
      }
      if( std::is_same< Device, Devices::Cuda >::value )
      {
#ifdef HAVE_CUDA
         Array< ElementType, Device, IndexType >* kernel_u =
                  Devices::Cuda::passToDevice( u );
         testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
         Devices::Cuda::freeFromDevice( kernel_u );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( u.getElement( i ) == i );
   };

   void testComparisonOperator()
   {
       using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > u;
      Array< ElementType, Device, IndexType > v;
      Array< ElementType, Device, IndexType > w;
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
      using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > u;
      Array< ElementType, Device, IndexType > v;
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
      Array< ElementType, Devices::Host, IndexType > w;
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
      using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > u;
      const int maxSize = 10;
      for( int i = 0; i < maxSize; i ++ )
         u. setSize( i );

      CPPUNIT_ASSERT( u. getSize() == maxSize - 1 );
   };

   void testReset()
   {
      using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > u;
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
      using namespace TNL::Containers;
      for( int i = 0; i < 100; i ++ )
      {
         Array< ElementType, Device, IndexType > u;
         u. setSize( i );
      }
   }

   void testSaveAndLoad()
   {
      using namespace TNL::Containers;
      Array< ElementType, Device, IndexType > v;
      v. setSize( 100 );
      for( int i = 0; i < 100; i ++ )
         v. setElement( i, 3.14147 );
      File file;
      file. open( "test-file.tnl", tnlWriteMode );
      v. save( file );
      file. close();
      Array< ElementType, Device, IndexType > u;
      file. open( "test-file.tnl", tnlReadMode );
      u. load( file );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }

   void testUnusualStructures()
   {
      using namespace TNL::Containers;
      Array< testingClassForArrayTester >u;
   };

};

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ElementType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getSize() )
      ( *u )[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

#endif /* HAVE_CPPUNIT */

#endif /* TNLARRAYTESTER_H_ */
