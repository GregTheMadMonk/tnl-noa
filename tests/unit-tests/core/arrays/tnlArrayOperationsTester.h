/***************************************************************************
                          tnlArrayOperationsTester.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLARRAYOPERATIONSTESTER_H_
#define TNLARRAYOPERATIONSTESTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

#include <core/arrays/tnlArrayOperations.h>
#include <core/tnlCuda.h>

template< typename Element, typename Device >
class tnlArrayOperationsTester{};


template< typename Element >
class tnlArrayOperationsTester< Element, tnlHost > : public CppUnit :: TestCase
{
   public:

      typedef tnlArrayOperationsTester< Element, tnlHost > ArrayOperationsTester;
      typedef CppUnit :: TestCaller< ArrayOperationsTester > TestCaller;

   tnlArrayOperationsTester(){};

   virtual
   ~tnlArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests->addTest( new TestCaller( "allocationTest", &ArrayOperationsTester::allocationTest ) );
      suiteOfTests->addTest( new TestCaller( "memorySetTest", &ArrayOperationsTester::memorySetTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryTest", &ArrayOperationsTester::copyMemoryTest ) );
      suiteOfTests->addTest( new TestCaller( "compareMemoryTest", &ArrayOperationsTester::compareMemoryTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryWithConversionTest", &ArrayOperationsTester::copyMemoryWithConversionTest ) );
      return suiteOfTests;
    };

    int getTestSize()
    {
       return 1 << 18;
       //const int cudaGridSize = 256;
       //return 1.5 * cudaGridSize * maxCudaBlockSize;
       //return  1 << 22;
    };

    void allocationTest()
    {
       Element* data;
       tnlArrayOperations< tnlHost >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( data != 0 );

       tnlArrayOperations< tnlHost >::freeMemory( data );
    };

    void memorySetTest()
    {
       const int size = 1024;
       Element *data;
       tnlArrayOperations< tnlHost > :: allocateMemory( data, size );
       tnlArrayOperations< tnlHost > :: setMemory( data, 13, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data[ i ] == 13 );
       tnlArrayOperations< tnlHost > :: freeMemory( data );
    };

    void copyMemoryTest()
    {
       const int size = getTestSize();

       Element *data1, *data2;
       tnlArrayOperations< tnlHost > :: allocateMemory( data1, size );
       tnlArrayOperations< tnlHost > :: allocateMemory( data2, size );
       tnlArrayOperations< tnlHost > :: setMemory( data1, 13, size );
       tnlArrayOperations< tnlHost > :: copyMemory< Element, Element, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ]);
       tnlArrayOperations< tnlHost > :: freeMemory( data1 );
       tnlArrayOperations< tnlHost > :: freeMemory( data2 );
    };

    void copyMemoryWithConversionTest()
    {
       const int size = getTestSize();
       int *data1;
       float *data2;
       tnlArrayOperations< tnlHost > :: allocateMemory( data1, size );
       tnlArrayOperations< tnlHost > :: allocateMemory( data2, size );
       tnlArrayOperations< tnlHost > :: setMemory( data1, 13, size );
       tnlArrayOperations< tnlHost > :: copyMemory< float, int, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ] );
       tnlArrayOperations< tnlHost > :: freeMemory( data1 );
       tnlArrayOperations< tnlHost > :: freeMemory( data2 );
    };


    void compareMemoryTest()
    {
       const int size = getTestSize();
       int *data1, *data2;
       tnlArrayOperations< tnlHost > :: allocateMemory( data1, size );
       tnlArrayOperations< tnlHost > :: allocateMemory( data2, size );
       tnlArrayOperations< tnlHost > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost > :: compareMemory< int, int, int >( data1, data2, size ) ) );
       tnlArrayOperations< tnlHost > :: setMemory( data2, 7, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost > :: compareMemory< int, int, int >( data1, data2, size ) ) );
    };

    void compareMemoryWithConversionTest()
    {
       const int size = getTestSize();
       int *data1;
       float *data2;
       tnlArrayOperations< tnlHost > :: allocateMemory( data1, size );
       tnlArrayOperations< tnlHost > :: allocateMemory( data2, size );
       tnlArrayOperations< tnlHost > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost > :: compareMemory< int, float, int >( data1, data2, size ) ) );
       tnlArrayOperations< tnlHost > :: setMemory( data2, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost > :: compareMemory< int, float, int >( data1, data2, size ) ) );
    };
};

template< typename Element >
class tnlArrayOperationsTester< Element, tnlCuda > : public CppUnit :: TestCase
{
   public:
      typedef tnlArrayOperationsTester< Element, tnlCuda > ArrayOperationsTester;
      typedef CppUnit :: TestCaller< ArrayOperationsTester > TestCaller;

   tnlArrayOperationsTester(){};

   virtual
   ~tnlArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests->addTest( new TestCaller( "allocationTest", &ArrayOperationsTester::allocationTest ) );
      suiteOfTests->addTest( new TestCaller( "setMemoryElementTest", &ArrayOperationsTester::setMemoryElementTest ) );
      suiteOfTests->addTest( new TestCaller( "getMemoryElementTest", &ArrayOperationsTester ::getMemoryElementTest ) );
      suiteOfTests->addTest( new TestCaller( "smallMemorySetTest", &ArrayOperationsTester::smallMemorySetTest ) );
      suiteOfTests->addTest( new TestCaller( "bigMemorySetTest", &ArrayOperationsTester::bigMemorySetTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryTest", &ArrayOperationsTester::copyMemoryTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryWithConversionHostToCudaTest", &ArrayOperationsTester::copyMemoryWithConversionHostToCudaTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryWithConversionCudaToHostTest", &ArrayOperationsTester::copyMemoryWithConversionCudaToHostTest ) );
      suiteOfTests->addTest( new TestCaller( "copyMemoryWithConversionCudaToCudaTest", &ArrayOperationsTester::copyMemoryWithConversionCudaToCudaTest ) );
      suiteOfTests->addTest( new TestCaller( "compareMemoryHostCudaTest", &ArrayOperationsTester::compareMemoryHostCudaTest ) );
      suiteOfTests->addTest( new TestCaller( "compareMemoryWithConevrsionHostCudaTest", &ArrayOperationsTester::compareMemoryWithConversionHostCudaTest ) );
      return suiteOfTests;
   }

    int getTestSize()
    {
       return 10000; //1 << 18;
       //const int cudaGridSize = 256;
       //return 1.5 * cudaGridSize * maxCudaBlockSize;
       //return  1 << 22;
    }

    void allocationTest()
    {
       int* data;
       tnlArrayOperations< tnlCuda >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( checkCudaDevice );

       tnlArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }

    void setMemoryElementTest()
    {
#ifdef HAVE_CUDA
       const int size( 1024 );
       int* data;
       tnlArrayOperations< tnlCuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          tnlArrayOperations< tnlCuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
       {
          int d;
          CPPUNIT_ASSERT( cudaMemcpy( &d, &data[ i ], sizeof( int ), cudaMemcpyDeviceToHost ) == cudaSuccess );
          CPPUNIT_ASSERT( d == i );
       }

       tnlArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
#endif
    }

    void getMemoryElementTest()
    {
       const int size( 1024 );
       int* data;
       tnlArrayOperations< tnlCuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          tnlArrayOperations< tnlCuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
          CPPUNIT_ASSERT( ( tnlArrayOperations< tnlCuda >::getMemoryElement( &data[ i ] ) == i ) );

       tnlArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }


    void smallMemorySetTest()
    {
       const int size = 1024;
       int *hostData, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData, 0, size );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       tnlArrayOperations< tnlCuda >::freeMemory( hostData );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void bigMemorySetTest()
    {
       const int size( getTestSize() );
       int *hostData, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData, 0, size );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i += 100 )
       {
          if( hostData[ i ] != 13 )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       }
       tnlArrayOperations< tnlHost >::freeMemory( hostData );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryTest()
    {
       const int size = getTestSize();

       int *hostData1, *hostData2, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       tnlArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData2, deviceData, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost >::compareMemory< int, int >( hostData1, hostData2, size) ) );
       tnlArrayOperations< tnlHost >::freeMemory( hostData1 );
       tnlArrayOperations< tnlHost >::freeMemory( hostData2 );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionHostToCudaTest()
    {
       const int size = getTestSize();
       int *hostData1;
       float *hostData2, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       tnlArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< float, int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< float, float, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       tnlArrayOperations< tnlHost >::freeMemory( hostData1 );
       tnlArrayOperations< tnlHost >::freeMemory( hostData2 );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToHostTest()
    {
       const int size = getTestSize();
       int *hostData1, *deviceData;
       float *hostData2;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       tnlArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< float, int, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       tnlArrayOperations< tnlHost >::freeMemory( hostData1 );
       tnlArrayOperations< tnlHost >::freeMemory( hostData2 );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToCudaTest()
    {
       const int size = getTestSize();
       int *hostData1, *deviceData1;
       float *hostData2, *deviceData2;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       tnlArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData1, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData2, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       tnlArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int, int >( deviceData1, hostData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< float, int, int >( deviceData2, deviceData1, size );
       tnlArrayOperations< tnlHost, tnlCuda >::copyMemory< float, float, int >( hostData2, deviceData2, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       tnlArrayOperations< tnlHost >::freeMemory( hostData1 );
       tnlArrayOperations< tnlHost >::freeMemory( hostData2 );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData1 );
       tnlArrayOperations< tnlCuda >::freeMemory( deviceData2 );
    };

    void compareMemoryHostCudaTest()
    {
       const int size = getTestSize();
       int *hostData, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData, 7, size );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 8, size );
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 7, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
    };

    void compareMemoryWithConversionHostCudaTest()
    {
       const int size = getTestSize();
       int *hostData;
       float *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData, 7, size );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, ( float ) 8.0, size );
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost, tnlCuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
    };
};


#else
template< typename Element, typename Device >
class tnlArrayOperationsTester{};
#endif /* HAVE_CPPUNIT */
#endif /* TNLARRAYOPERATIONSTESTER_H_ */
