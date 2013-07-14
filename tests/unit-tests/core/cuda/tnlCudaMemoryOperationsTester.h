/***************************************************************************
                          tnlCudaMemoryOperationsTester.h  -  description
                             -------------------
    begin                : Mar 20, 2013
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

#ifndef TNLCUDAMEMORYOPERATIONSTESTER_H_
#define TNLCUDAMEMORYOPERATIONSTESTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/cuda/device-check.h>
#include <implementation/core/memory-operations.h>

class tnlCudaMemoryOperationsTester : public CppUnit :: TestCase
{
   public:
   tnlCudaMemoryOperationsTester(){};

   virtual
   ~tnlCudaMemoryOperationsTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCudaMemoryOperationsTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "allocationTest",
                                &tnlCudaMemoryOperationsTester :: allocationTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "smallMemorySetTest",
                                &tnlCudaMemoryOperationsTester :: smallMemorySetTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "bigMemorySetTest",
                                &tnlCudaMemoryOperationsTester :: bigMemorySetTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "copyMemoryTest",
                                &tnlCudaMemoryOperationsTester :: copyMemoryTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "copyMemoryWithConversionHostToCudaTest",
                                &tnlCudaMemoryOperationsTester :: copyMemoryWithConversionHostToCudaTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "copyMemoryWithConversionCudaToHostTest",
                                &tnlCudaMemoryOperationsTester :: copyMemoryWithConversionCudaToHostTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "copyMemoryWithConversionCudaToCudaTest",
                                &tnlCudaMemoryOperationsTester :: copyMemoryWithConversionCudaToCudaTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "compareMemoryHostCudaTest",
                                &tnlCudaMemoryOperationsTester :: compareMemoryHostCudaTest )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaMemoryOperationsTester >(
                                "compareMemoryWithConevrsionHostCudaTest",
                                &tnlCudaMemoryOperationsTester :: compareMemoryWithConversionHostCudaTest )
                               );

      return suiteOfTests;
   };

   int getTestSize()
   {
      const int cudaGridSize = 256;
      return 1.5 * cudaGridSize * maxCudaBlockSize;
      //return  1 << 22;
   };

   void allocationTest()
   {
      int* data;
      allocateMemoryCuda( data, getTestSize() );
      CPPUNIT_ASSERT( checkCudaDevice );

      freeMemoryCuda( data );
      CPPUNIT_ASSERT( checkCudaDevice );
   };

   void smallMemorySetTest()
   {
      const int size = 1024;
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 0, size );
      setMemoryCuda( deviceData, 13, size, maxCudaGridSize );
      CPPUNIT_ASSERT( checkCudaDevice );
      copyMemoryCudaToHost( hostData, deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( hostData[ i ] == 13 );
      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
   };

   void bigMemorySetTest()
   {
      const int size( getTestSize() );
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 0, size );
      setMemoryCuda( deviceData, 13, size, maxCudaGridSize );
      CPPUNIT_ASSERT( checkCudaDevice );
      copyMemoryCudaToHost( hostData, deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      for( int i = 0; i < size; i += 100 )
      {
         if( hostData[ i ] != 13 )
         CPPUNIT_ASSERT( hostData[ i ] == 13 );
      }
      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
   };

   void copyMemoryTest()
   {
      const int size = getTestSize();

      int *hostData1, *hostData2, *deviceData;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData1, 13, size );
      copyMemoryHostToCuda( deviceData, hostData1, size );
      copyMemoryCudaToHost( hostData2, deviceData, size );
      CPPUNIT_ASSERT( compareMemoryHost( hostData1, hostData2, size) );
      freeMemoryHost( hostData1 );
      freeMemoryHost( hostData2 );
      freeMemoryCuda( deviceData );
   };

   void copyMemoryWithConversionHostToCudaTest()
   {
      const int size = getTestSize();
      int *hostData1;
      float *hostData2, *deviceData;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData1, 13, size );
      copyMemoryHostToCuda( deviceData, hostData1, size );
      copyMemoryCudaToHost( hostData2, deviceData, size );
      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
      freeMemoryHost( hostData1 );
      freeMemoryHost( hostData2 );
      freeMemoryCuda( deviceData );
   };

   void copyMemoryWithConversionCudaToHostTest()
   {
      const int size = getTestSize();
      int *hostData1, *deviceData;
      float *hostData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData1, 13, size );
      copyMemoryHostToCuda( deviceData, hostData1, size );
      copyMemoryCudaToHost( hostData2, deviceData, size );
      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
      freeMemoryHost( hostData1 );
      freeMemoryHost( hostData2 );
      freeMemoryCuda( deviceData );
   };

   void copyMemoryWithConversionCudaToCudaTest()
   {
      const int size = getTestSize();
      int *hostData1, *deviceData1;
      float *hostData2, *deviceData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData1, size );
      allocateMemoryCuda( deviceData2, size );
      setMemoryHost( hostData1, 13, size );
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      copyMemoryCudaToCuda( deviceData2, deviceData1, size, maxCudaGridSize );
      copyMemoryCudaToHost( hostData2, deviceData2, size );
      for( int i = 0; i < size; i ++ )
         CPPUNIT_ASSERT( hostData1[ i ] == hostData2[ i ] );
      freeMemoryHost( hostData1 );
      freeMemoryHost( hostData2 );
      freeMemoryCuda( deviceData1 );
      freeMemoryCuda( deviceData2 );
   };

   void compareMemoryHostCudaTest()
   {
      const int size = getTestSize();
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 7, size );
      setMemoryCuda( deviceData, 8, size, maxCudaGridSize );
      CPPUNIT_ASSERT( ! compareMemoryHostCuda( hostData, deviceData, size ) );
      setMemoryCuda( deviceData, 7, size, maxCudaGridSize );
      CPPUNIT_ASSERT( compareMemoryHostCuda( hostData, deviceData, size ) );
   };

   void compareMemoryWithConversionHostCudaTest()
   {
      const int size = getTestSize();
      int *hostData;
      float *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 7, size );
      setMemoryCuda( deviceData, ( float ) 8.0, size, maxCudaGridSize );
      CPPUNIT_ASSERT( ! compareMemoryHostCuda( hostData, deviceData, size ) );
      setMemoryCuda( deviceData, ( float ) 7.0, size, maxCudaGridSize );
      CPPUNIT_ASSERT( compareMemoryHostCuda( hostData, deviceData, size ) );
   };


};

#else
class tnlCudaMemoryOperationsTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLCUDAMEMORYOPERATIONSTESTER_H_ */
