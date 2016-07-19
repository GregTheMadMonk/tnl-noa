/***************************************************************************
                          tnlMultiArrayTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMULTIARRAYTESTER_H_
#define TNLMULTIARRAYTESTER_H_

#ifdef HAVE_CPPUNIT

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/arrays/tnlMultiArray.h>
#include <core/tnlFile.h>

using namespace TNL;

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( tnlMultiArray< 1, ElementType, tnlCuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() )
      ( *u )( threadIdx.x ) = threadIdx.x;
}

template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( tnlMultiArray< 2, ElementType, tnlCuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() &&
       threadIdx.x < ( *u ).getDimensions().y() )
      ( *u )( threadIdx.x, threadIdx.x ) = threadIdx.x;
}

template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( tnlMultiArray< 3, ElementType, tnlCuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() &&
       threadIdx.x < ( *u ).getDimensions().y() &&
       threadIdx.x < ( *u ).getDimensions().z() )
      ( *u )( threadIdx.x, threadIdx.x, threadIdx.x ) = threadIdx.x;
}

#endif /* HAVE_CUDA */

template< int Dimensions, typename ElementType, typename Device, typename IndexType >
class tnlMultiArrayTester : public CppUnit :: TestCase
{
   public:

   typedef tnlMultiArrayTester< Dimensions, ElementType, Device, IndexType > MultiArrayTester;
   typedef CppUnit :: TestCaller< MultiArrayTester > TestCaller;

   tnlMultiArrayTester(){};

   virtual
   ~tnlMultiArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMultiArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testConstructorDestructor", &MultiArrayTester::testConstructorDestructor ) );
      suiteOfTests -> addTest( new TestCaller( "testSetSize", &MultiArrayTester::testSetSize ) );
      suiteOfTests -> addTest( new TestCaller( "testSetGetElement", &MultiArrayTester::testSetGetElement ) );
      suiteOfTests -> addTest( new TestCaller( "testComparisonOperator", &MultiArrayTester::testComparisonOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testEquivalenceOperator", &MultiArrayTester::testEquivalenceOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testGetSize", &MultiArrayTester::testGetSize ) );
      suiteOfTests -> addTest( new TestCaller( "testReset", &MultiArrayTester::testReset ) );
      suiteOfTests -> addTest( new TestCaller( "testSetSizeAndDestructor", &MultiArrayTester::testSetSizeAndDestructor ) );
      suiteOfTests -> addTest( new TestCaller( "testSaveAndLoad", &MultiArrayTester::testSaveAndLoad ) );
      return suiteOfTests;
   }

   void testConstructorDestructor()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
   }

   void testSetSize()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u, v;
      u. setDimensions( 10 );
      v. setDimensions( 10 );
   }

   void setDiagonalElement( tnlMultiArray< 1, ElementType, Device, IndexType >& u,
                            const IndexType& i,
                            const ElementType& v )
   {
      u.setElement( i, v );
   }

   void setDiagonalElement( tnlMultiArray< 2, ElementType, Device, IndexType >& u,
                            const IndexType& i,
                            const ElementType& v )
   {
      u.setElement( i, i, v );
   }

   void setDiagonalElement( tnlMultiArray< 3, ElementType, Device, IndexType >& u,
                            const IndexType& i,
                            const ElementType& v )
   {
      u.setElement( i, i, i, v );
   }
 
   IndexType getDiagonalElement( tnlMultiArray< 1, ElementType, Device, IndexType >& u,
                                 const IndexType& i )
   {
      return u.getElement( i );
   }
 
   IndexType getDiagonalElement( tnlMultiArray< 2, ElementType, Device, IndexType >& u,
                                 const IndexType& i )
   {
      return u.getElement( i, i );
   }
 
   IndexType getDiagonalElement( tnlMultiArray< 3, ElementType, Device, IndexType >& u,
                                 const IndexType& i )
   {
      return u.getElement( i, i, i );
   }


   void testSetGetElement()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
      u. setDimensions( 10 );
      if( Device::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 10; i ++ )
            this->setDiagonalElement( u, i, i  );
      }
      if( Device::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         tnlMultiArray< Dimensions, ElementType, Device, IndexType >* kernel_u =
                  tnlCuda::passToDevice( u );
         testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
         tnlCuda::freeFromDevice( kernel_u );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( getDiagonalElement( u, i ) == i );
   };

   void testComparisonOperator()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u, v, w;
      u.setDimensions( 10 );
      v.setDimensions( 10 );
      w.setDimensions( 10 );
      u.setValue( 0 );
      v.setValue( 0 );
      w.setValue( 0 );
      for( int i = 0; i < 10; i ++ )
      {
         setDiagonalElement( u, i, i );
         setDiagonalElement( v, i, i );
         setDiagonalElement( w, i, 2*1 );
      }
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
      CPPUNIT_ASSERT( u != w );
      CPPUNIT_ASSERT( ! ( u == w ) );
   };

   void testEquivalenceOperator()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > v;
      u. setDimensions( 10 );
      v. setDimensions( 10 );
      for( int i = 0; i < 10; i ++ )
         setDiagonalElement( u, i, i );
      v = u;
      CPPUNIT_ASSERT( u == v );
      CPPUNIT_ASSERT( ! ( u != v ) );
   };

   void testGetSize()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
      const int maxSize = 10;
      for( int i = 1; i < maxSize; i ++ )
         u. setDimensions( i );

      CPPUNIT_ASSERT( u. getDimensions().x() == maxSize - 1 );
   };

   void testReset()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
      u.setDimensions( 100 );
      CPPUNIT_ASSERT( u. getDimensions().x() == 100 );
      u.reset();
      CPPUNIT_ASSERT( u. getDimensions().x() == 0 );
      u.setDimensions( 100 );
      CPPUNIT_ASSERT( u. getDimensions().x() == 100 );
      u.reset();
      CPPUNIT_ASSERT( u. getDimensions().x() == 0 );

   };

   void testSetSizeAndDestructor()
   {
      for( int i = 1; i < 100; i ++ )
      {
         tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
         u. setDimensions( i );
      }
   }

   void testSaveAndLoad()
   {
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > v;
      const int size( 10 );
      CPPUNIT_ASSERT( v. setDimensions( size ) );
      for( int i = 0; i < size; i ++ )
         setDiagonalElement( v, i, 3.14147 );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode );
      CPPUNIT_ASSERT( v. save( file ) );
      file. close();
      tnlMultiArray< Dimensions, ElementType, Device, IndexType > u;
      file. open( "test-file.tnl", tnlReadMode );
      CPPUNIT_ASSERT( u. load( file ) );
      file. close();
      CPPUNIT_ASSERT( u == v );
   }
};

#endif /* HAVE_CPPUNIT */

#endif /* TNLMULTIARRAYTESTER_H_ */
