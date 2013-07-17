/***************************************************************************
                          tnlArrayOperationsTester.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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
#include <core/cuda/device-check.h>

template< typename Element, typename Device >
class tnlArrayOperationsTester{};


template< typename Element >
class tnlArrayOperationsTester< Element, tnlHost > : public CppUnit :: TestCase
{
   public:
   tnlArrayOperationsTester(){};

   virtual
   ~tnlArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "allocationTest",
                                 &tnlArrayOperationsTester :: allocationTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "memorySetTest",
                                 &tnlArrayOperationsTester :: memorySetTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryTest",
                                 &tnlArrayOperationsTester :: copyMemoryTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "compareMemoryTest",
                                 &tnlArrayOperationsTester :: compareMemoryTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryWithConversionTest",
                                 &tnlArrayOperationsTester :: copyMemoryWithConversionTest )
                                );
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
       tnlArrayOperations< tnlHost > :: copyMemory< Element, tnlHost, Element, int >( data2, data1, size );
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
       tnlArrayOperations< tnlHost > :: copyMemory< float, tnlHost, int, int >( data2, data1, size );
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
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost > :: compareMemory< int, tnlHost, int, int >( data1, data2, size ) ) );
       tnlArrayOperations< tnlHost > :: setMemory( data2, 7, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost > :: compareMemory< int, tnlHost, int, int >( data1, data2, size ) ) );
    };

    void compareMemoryWithConversionTest()
    {
       const int size = getTestSize();
       int *data1;
       float *data2;
       tnlArrayOperations< tnlHost > :: allocateMemory( data1, size );
       tnlArrayOperations< tnlHost > :: allocateMemory( data2, size );
       tnlArrayOperations< tnlHost > :: setMemory( data1, 7, size );
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlHost > :: compareMemory< int, tnlHost, float, int >( data1, data2, size ) ) );
       tnlArrayOperations< tnlHost > :: setMemory( data2, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost > :: compareMemory< int, tnlHost, float, int >( data1, data2, size ) ) );
    };
};

template< typename Element >
class tnlArrayOperationsTester< Element, tnlCuda > : public CppUnit :: TestCase
{
   public:
   tnlArrayOperationsTester(){};

   virtual
   ~tnlArrayOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "allocationTest",
                                 &tnlArrayOperationsTester :: allocationTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "smallMemorySetTest",
                                 &tnlArrayOperationsTester :: smallMemorySetTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "bigMemorySetTest",
                                 &tnlArrayOperationsTester :: bigMemorySetTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryTest",
                                 &tnlArrayOperationsTester :: copyMemoryTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryWithConversionHostToCudaTest",
                                 &tnlArrayOperationsTester :: copyMemoryWithConversionHostToCudaTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryWithConversionCudaToHostTest",
                                 &tnlArrayOperationsTester :: copyMemoryWithConversionCudaToHostTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "copyMemoryWithConversionCudaToCudaTest",
                                 &tnlArrayOperationsTester :: copyMemoryWithConversionCudaToCudaTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "compareMemoryHostCudaTest",
                                 &tnlArrayOperationsTester :: compareMemoryHostCudaTest )
                                );
       suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayOperationsTester >(
                                 "compareMemoryWithConevrsionHostCudaTest",
                                 &tnlArrayOperationsTester :: compareMemoryWithConversionHostCudaTest )
                                );

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
       int* data;
       tnlArrayOperations< tnlCuda >::allocateMemory( data, getTestSize() );
       CPPUNIT_ASSERT( checkCudaDevice );

       tnlArrayOperations< tnlCuda >::freeMemory( data );
       CPPUNIT_ASSERT( checkCudaDevice );
    };

    void smallMemorySetTest()
    {
       const int size = 1024;
       int *hostData, *deviceData;
       tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
       tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
       tnlArrayOperations< tnlHost >::setMemory( hostData, 0, size );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 13, size );
       CPPUNIT_ASSERT( checkCudaDevice );
       tnlArrayOperations< tnlCuda >::copyMemory< int, tnlHost, int, int >( hostData, deviceData, size );
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
       tnlArrayOperations< tnlCuda >::copyMemory< int, tnlHost, int, int >( hostData, deviceData, size );
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
       tnlArrayOperations< tnlHost >::copyMemory< int, tnlCuda, int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< int, tnlHost, int, int >( hostData2, deviceData, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlHost >::compareMemory< int, tnlHost, int, int >( hostData1, hostData2, size) ) );
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
       tnlArrayOperations< tnlHost >::copyMemory< float, tnlCuda, int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< float, tnlHost, float, int >( hostData2, deviceData, size );
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
       tnlArrayOperations< tnlHost >::copyMemory< int, tnlCuda, int, int >( deviceData, hostData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< float, tnlHost, int, int >( hostData2, deviceData, size );
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
       tnlArrayOperations< tnlHost >::copyMemory< int, tnlCuda, int, int >( deviceData1, hostData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< float, tnlCuda, int, int >( deviceData2, deviceData1, size );
       tnlArrayOperations< tnlCuda >::copyMemory< float, tnlHost, float, int >( hostData2, deviceData2, size );
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
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlCuda >::compareMemory< int, tnlHost, int, int >( hostData, deviceData, size ) ) );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, 7, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlCuda >::compareMemory< int, tnlHost, int, int >( hostData, deviceData, size ) ) );
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
       CPPUNIT_ASSERT( ( ! tnlArrayOperations< tnlCuda >::compareMemory< int, tnlHost, float, int >( hostData, deviceData, size ) ) );
       tnlArrayOperations< tnlCuda >::setMemory( deviceData, ( float ) 7.0, size );
       CPPUNIT_ASSERT( ( tnlArrayOperations< tnlCuda >::compareMemory< int, tnlHost, float, int >( hostData, deviceData, size ) ) );
    };
};


#else
template< typename Element, typename Device >
class tnlArrayOperationsTester{};
#endif /* HAVE_CPPUNIT */
#endif /* TNLARRAYOPERATIONSTESTER_H_ */
