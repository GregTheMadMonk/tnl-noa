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
                                "copyTest",
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

      return suiteOfTests;
   };

   void allocationTest()
   {
      int* data;
      allocateMemoryCuda( data, 100 );
      CPPUNIT_ASSERT( checkCudaDevice );

      freeMemoryCuda( data );
      CPPUNIT_ASSERT( checkCudaDevice );
   };

   void copyTest()
   {
      const int size( 1 << 22 );
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

   void smallMemorySetTest()
   {
      const int size( 1024 );
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 0, size );
      setMemoryCuda( deviceData, 13, size );
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
      const int size( 1.1 * maxCudaGridSize * maxCudaBlockSize );
      cout << "Size = " << size << endl;
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 0, size );
      setMemoryCuda( deviceData, 13, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      copyMemoryCudaToHost( hostData, deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      for( int i = 0; i < size; i += 100 )
      {
         if( hostData[ i ] != 13 )
            cout << " i = " << i << " " << hostData[ i ] << endl;
         CPPUNIT_ASSERT( hostData[ i ] == 13 );
      }
      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
   };

};

#else
class tnlCudaMemoryOperationsTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLCUDAMEMORYOPERATIONSTESTER_H_ */
