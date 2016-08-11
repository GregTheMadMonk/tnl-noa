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

#include <TNL/Containers/ArrayOperations.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;

template< typename Element, typename Device >
class ArrayOperationsTester{};


template< typename Element >
class ArrayOperationsTester< Element, Devices::Host > : public CppUnit :: TestCase
{
   public:

      typedef ArrayOperationsTester< Element, Devices::Host > ArrayOperationsTesterType;
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
       using namespace TNL::Containers;
       Element* data;
       ArrayOperations< Devices::Host >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( data != 0 );

       ArrayOperations< Devices::Host >::freeMemory( data );
    };

    void memorySetTest()
    {
       using namespace TNL::Containers;
       const int size = 1024;
       Element *data;
       ArrayOperations< Devices::Host > :: allocateMemory( data, size );
       ArrayOperations< Devices::Host > :: setMemory( data, 13, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data[ i ] == 13 );
       ArrayOperations< Devices::Host > :: freeMemory( data );
    };

    void copyMemoryTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();

       Element *data1, *data2;
       ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
       ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
       ArrayOperations< Devices::Host > :: setMemory( data1, 13, size );
       ArrayOperations< Devices::Host > :: copyMemory< Element, Element, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ]);
       ArrayOperations< Devices::Host > :: freeMemory( data1 );
       ArrayOperations< Devices::Host > :: freeMemory( data2 );
    };

    void copyMemoryWithConversionTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *data1;
       float *data2;
       ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
       ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
       ArrayOperations< Devices::Host > :: setMemory( data1, 13, size );
       ArrayOperations< Devices::Host > :: copyMemory< float, int, int >( data2, data1, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( data1[ i ] == data2[ i ] );
       ArrayOperations< Devices::Host > :: freeMemory( data1 );
       ArrayOperations< Devices::Host > :: freeMemory( data2 );
    };


    void compareMemoryTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *data1, *data2;
       ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
       ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
       ArrayOperations< Devices::Host > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< Devices::Host > :: compareMemory< int, int, int >( data1, data2, size ) ) );
       ArrayOperations< Devices::Host > :: setMemory( data2, 7, size );
       CPPUNIT_ASSERT( ( ArrayOperations< Devices::Host > :: compareMemory< int, int, int >( data1, data2, size ) ) );
    };

    void compareMemoryWithConversionTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *data1;
       float *data2;
       ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
       ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
       ArrayOperations< Devices::Host > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< Devices::Host > :: compareMemory< int, float, int >( data1, data2, size ) ) );
       ArrayOperations< Devices::Host > :: setMemory( data2, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( ArrayOperations< Devices::Host > :: compareMemory< int, float, int >( data1, data2, size ) ) );
    };
};

template< typename Element >
class ArrayOperationsTester< Element, Devices::Cuda > : public CppUnit :: TestCase
{
   public:
      typedef ArrayOperationsTester< Element, Devices::Cuda > ArrayOperationsTesterType;
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
       using namespace TNL::Containers;
       int* data;
       ArrayOperations< Devices::Cuda >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( checkCudaDevice );

       ArrayOperations< Devices::Cuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }

    void setMemoryElementTest()
    {
#ifdef HAVE_CUDA
       using namespace TNL::Containers;
       const int size( 1024 );
       int* data;
       ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          ArrayOperations< Devices::Cuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
       {
          int d;
          CPPUNIT_ASSERT( cudaMemcpy( &d, &data[ i ], sizeof( int ), cudaMemcpyDeviceToHost ) == cudaSuccess );
          CPPUNIT_ASSERT( d == i );
       }

       ArrayOperations< Devices::Cuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
#endif
    }

    void getMemoryElementTest()
    {
       using namespace TNL::Containers;
       const int size( 1024 );
       int* data;
       ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
       CPPUNIT_ASSERT( checkCudaDevice );

       for( int i = 0; i < getTestSize(); i++ )
          ArrayOperations< Devices::Cuda >::setMemoryElement( &data[ i ], i );

       for( int i = 0; i < size; i++ )
          CPPUNIT_ASSERT( ( ArrayOperations< Devices::Cuda >::getMemoryElement( &data[ i ] ) == i ) );

       ArrayOperations< Devices::Cuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    }


    void smallMemorySetTest()
    {
       using namespace TNL::Containers;
       const int size = 1024;
       int *hostData, *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData, 0, size );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       ArrayOperations< Devices::Cuda >::freeMemory( hostData );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
    };

    void bigMemorySetTest()
    {
       using namespace TNL::Containers;
       const int size( getTestSize() );
       int *hostData, *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData, 0, size );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData, deviceData, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       for( int i = 0; i < size; i += 100 )
       {
          if( hostData[ i ] != 13 )
          CPPUNIT_ASSERT( hostData[ i ] == 13 );
       }
       ArrayOperations< Devices::Host >::freeMemory( hostData );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
    };

    void copyMemoryTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();

       int *hostData1, *hostData2, *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
       ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
       ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int >( deviceData, hostData1, size );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData2, deviceData, size );
       CPPUNIT_ASSERT( ( ArrayOperations< Devices::Host >::compareMemory< int, int >( hostData1, hostData2, size) ) );
       ArrayOperations< Devices::Host >::freeMemory( hostData1 );
       ArrayOperations< Devices::Host >::freeMemory( hostData2 );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionHostToCudaTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *hostData1;
       float *hostData2, *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
       ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
       ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float, int, int >( deviceData, hostData1, size );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, float, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< Devices::Host >::freeMemory( hostData1 );
       ArrayOperations< Devices::Host >::freeMemory( hostData2 );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToHostTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *hostData1, *deviceData;
       float *hostData2;
       ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
       ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
       ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int >( deviceData, hostData1, size );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, int, int >( hostData2, deviceData, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< Devices::Host >::freeMemory( hostData1 );
       ArrayOperations< Devices::Host >::freeMemory( hostData2 );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
    };

    void copyMemoryWithConversionCudaToCudaTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *hostData1, *deviceData1;
       float *hostData2, *deviceData2;
       ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
       ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData1, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );
       ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
       ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int, int >( deviceData1, hostData1, size );
       ArrayOperations< Devices::Cuda >::copyMemory< float, int, int >( deviceData2, deviceData1, size );
       ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, float, int >( hostData2, deviceData2, size );
       for( int i = 0; i < size; i ++ )
          CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
       ArrayOperations< Devices::Host >::freeMemory( hostData1 );
       ArrayOperations< Devices::Host >::freeMemory( hostData2 );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData1 );
       ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
    };

    void compareMemoryHostCudaTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *hostData, *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData, 7, size );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, 8, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, 7, size );
       CPPUNIT_ASSERT( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
    };

    void compareMemoryWithConversionHostCudaTest()
    {
       using namespace TNL::Containers;
       const int size = getTestSize();
       int *hostData;
       float *deviceData;
       ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
       ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
       ArrayOperations< Devices::Host >::setMemory( hostData, 7, size );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, ( float ) 8.0, size );
       CPPUNIT_ASSERT( ( ! ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
       ArrayOperations< Devices::Cuda >::setMemory( deviceData, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
    };
};


#else
template< typename Element, typename Device >
class ArrayOperationsTester{};
#endif /* HAVE_CPPUNIT */
#endif /* TNLARRAYOPERATIONSTESTER_H_ */
