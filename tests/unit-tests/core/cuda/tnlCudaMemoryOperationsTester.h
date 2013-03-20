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

   void smallMemorySetTest()
   {
      const int size( 100 );
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
   };

   void bigMemorySetTest()
   {
      const int size( 2.7 * maxCudaGridSize * maxCudaBlockSize );
      cout << "Size = " << size << endl;
      int *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      setMemoryHost( hostData, 0, size );
      setMemoryCuda( deviceData, 13, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      copyMemoryCudaToHost( hostData, deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      for( int i = 0; i < size; i ++ )
      {
         if( hostData[ i ] != 13 )
            cout << " i = " << i << endl;
         CPPUNIT_ASSERT( hostData[ i ] == 13 );
      }
   };

};



#endif /* TNLCUDAMEMORYOPERATIONSTESTER_H_ */
