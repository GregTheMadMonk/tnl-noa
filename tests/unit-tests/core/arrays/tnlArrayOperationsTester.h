/***************************************************************************
                          ArrayOperationsTester.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLARRAYOPERATIONSTESTER_H_
#define TNLARRAYOPERATIONSTESTER_H_

#include <TNL/tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

#include <TNL/Arrays/ArrayOperations.h>
#include <TNL/core/tnlCuda.h>

using namespace TNL;

template< typename Element, typename Device >
class ArrayOperationsTester{};


template< typename Element >
class ArrayOperationsTester< Element, tnlHost > : public CppUnit :: TestCase
{
   public:

      typedef ArrayOperationsTester< Element, tnlHost > ArrayOperationsTesterType;
      typedef CppUnit :: TestCaller< ArrayOperationsTester > TestCaller;

   ArrayOperationsTester(){};

   virtual
   ~ArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "ArrayOperationsTester" );
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
       using namespace TNL::Arrays;
       Element* data;
       ArrayOperations< tnlHost >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( data != 0 );

       ArrayOperations< tnlHost >::freeMemory( data );
    };

    void memorySetTest()
    {
       using namespace TNL::Arrays;
       const int size = 1024;
       Element *data;
       ArrayOperations< tnlHost > :: allocateMemory( data, size );
       ArrayOperations< tnlHost > :: setMemory( data, 13, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data[ i ] == 13 );
       ArrayOperations< tnlHost > :: freeMemory( data );
    };

    void copyMemoryTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();

       Element *data1, *data2;
       ArrayOperations< tnlHost > :: allocateMemory( data1, size );
       ArrayOperations< tnlHost > :: allocateMemory( data2, size );
       ArrayOperations< tnlHost > :: setMemory( data1, 13, size );
       ArrayOperations< tnlHost > :: copyMemory< Element, Element, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ]);
       ArrayOperations< tnlHost > :: freeMemory( data1 );
       ArrayOperations< tnlHost > :: freeMemory( data2 );
    };

    void copyMemoryWithConversionTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *data1;
       float *data2;
       ArrayOperations< tnlHost > :: allocateMemory( data1, size );
       ArrayOperations< tnlHost > :: allocateMemory( data2, size );
       ArrayOperations< tnlHost > :: setMemory( data1, 13, size );
       ArrayOperations< tnlHost > :: copyMemory< float, int, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ] );
       ArrayOperations< tnlHost > :: freeMemory( data1 );
       ArrayOperations< tnlHost > :: freeMemory( data2 );
    };


    void compareMemoryTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *data1, *data2;
       ArrayOperations< tnlHost > :: allocateMemory( data1, size );
       ArrayOperations< tnlHost > :: allocateMemory( data2, size );
       ArrayOperations< tnlHost > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< tnlHost > :: compareMemory< int, int, int >( data1, data2, size ) ) );
       ArrayOperations< tnlHost > :: setMemory( data2, 7, size );
       CPPUNIT_ASSERT( ( ArrayOperations< tnlHost > :: compareMemory< int, int, int >( data1, data2, size ) ) );
    };

    void compareMemoryWithConversionTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *data1;
       float *data2;
       ArrayOperations< tnlHost > :: allocateMemory( data1, size );
       ArrayOperations< tnlHost > :: allocateMemory( data2, size );
       ArrayOperations< tnlHost > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< tnlHost > :: compareMemory< int, float, int >( data1, data2, size ) ) );
       ArrayOperations< tnlHost > :: setMemory( data2, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( ArrayOperations< tnlHost > :: compareMemory< int, float, int >( data1, data2, size ) ) );
    };
};

template< typename Element >
class ArrayOperationsTester< Element, tnlCuda > : public CppUnit :: TestCase
{
   public:
      typedef ArrayOperationsTester< Element, tnlCuda > ArrayOperationsTesterType;
      typedef CppUnit :: TestCaller< ArrayOperationsTester > TestCaller;

   ArrayOperationsTester(){};

   virtual
   ~ArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "ArrayOperationsTester" );
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
       using namespace TNL::Arrays;
       int* data;
       ArrayOperations< tnlCuda >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( checkCudaDevice );

       ArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }

    void setMemoryElementTest()
    {
#ifdef HAVE_CUDA
       using namespace TNL::Arrays;
       const int size( 1024 );
       int* data;
       ArrayOperations< tnlCuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          ArrayOperations< tnlCuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
       {
          int d;
          CPPUNIT_ASSERT( cudaMemcpy( &d, &data[ i ], sizeof( int ), cudaMemcpyDeviceToHost ) == cudaSuccess );
          CPPUNIT_ASSERT( d == i );
       }

       ArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
#endif
    }

    void getMemoryElementTest()
    {
       using namespace TNL::Arrays;
       const int size( 1024 );
       int* data;
       ArrayOperations< tnlCuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          ArrayOperations< tnlCuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
          CPPUNIT_ASSERT( ( ArrayOperations< tnlCuda >::getMemoryElement( &data[ i ] ) == i ) );

       ArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }


    void smallMemorySetTest()
    {
       using namespace TNL::Arrays;
       const int size = 1024;
       int *hostData, *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData, 0, size );
       ArrayOperations< tnlCuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       ArrayOperations< tnlCuda >::freeMemory( hostData );
       ArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void bigMemorySetTest()
    {
       using namespace TNL::Arrays;
       const int size( getTestSize() );
       int *hostData, *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData, 0, size );
       ArrayOperations< tnlCuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i += 100 )
       {
          if( hostData[ i ] != 13 )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       }
       ArrayOperations< tnlHost >::freeMemory( hostData );
       ArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();

       int *hostData1, *hostData2, *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       ArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       ArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int >( deviceData, hostData1, size );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< int, int >( hostData2, deviceData, size );
       CPPUNIT_ASSERT( ( ArrayOperations< tnlHost >::compareMemory< int, int >( hostData1, hostData2, size) ) );
       ArrayOperations< tnlHost >::freeMemory( hostData1 );
       ArrayOperations< tnlHost >::freeMemory( hostData2 );
       ArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionHostToCudaTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *hostData1;
       float *hostData2, *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       ArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       ArrayOperations< tnlCuda, tnlHost >::copyMemory< float, int, int >( deviceData, hostData1, size );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< float, float, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< tnlHost >::freeMemory( hostData1 );
       ArrayOperations< tnlHost >::freeMemory( hostData2 );
       ArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToHostTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *hostData1, *deviceData;
       float *hostData2;
       ArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       ArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       ArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int >( deviceData, hostData1, size );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< float, int, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< tnlHost >::freeMemory( hostData1 );
       ArrayOperations< tnlHost >::freeMemory( hostData2 );
       ArrayOperations< tnlCuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToCudaTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *hostData1, *deviceData1;
       float *hostData2, *deviceData2;
       ArrayOperations< tnlHost >::allocateMemory( hostData1, size );
       ArrayOperations< tnlHost >::allocateMemory( hostData2, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData1, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData2, size );
       ArrayOperations< tnlHost >::setMemory( hostData1, 13, size );
       ArrayOperations< tnlCuda, tnlHost >::copyMemory< int, int, int >( deviceData1, hostData1, size );
       ArrayOperations< tnlCuda >::copyMemory< float, int, int >( deviceData2, deviceData1, size );
       ArrayOperations< tnlHost, tnlCuda >::copyMemory< float, float, int >( hostData2, deviceData2, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< tnlHost >::freeMemory( hostData1 );
       ArrayOperations< tnlHost >::freeMemory( hostData2 );
       ArrayOperations< tnlCuda >::freeMemory( deviceData1 );
       ArrayOperations< tnlCuda >::freeMemory( deviceData2 );
    };

    void compareMemoryHostCudaTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *hostData, *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData, 7, size );
       ArrayOperations< tnlCuda >::setMemory( deviceData, 8, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< tnlHost, tnlCuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
       ArrayOperations< tnlCuda >::setMemory( deviceData, 7, size );
       CPPUNIT_ASSERT( ( ArrayOperations< tnlHost, tnlCuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
    };

    void compareMemoryWithConversionHostCudaTest()
    {
       using namespace TNL::Arrays;
       const int size = getTestSize();
       int *hostData;
       float *deviceData;
       ArrayOperations< tnlHost >::allocateMemory( hostData, size );
       ArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       ArrayOperations< tnlHost >::setMemory( hostData, 7, size );
       ArrayOperations< tnlCuda >::setMemory( deviceData, ( float ) 8.0, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< tnlHost, tnlCuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
       ArrayOperations< tnlCuda >::setMemory( deviceData, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( ArrayOperations< tnlHost, tnlCuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
    };
};


#else
template< typename Element, typename Device >
class ArrayOperationsTester{};
#endif /* HAVE_CPPUNIT */
#endif /* TNLARRAYOPERATIONSTESTER_H_ */
