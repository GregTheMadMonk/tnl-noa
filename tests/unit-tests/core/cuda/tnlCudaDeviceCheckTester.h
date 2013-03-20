/***************************************************************************
                          tnlCudaDeviceCheckTester.h  -  description
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

#ifndef TNLCUDADEVICECHECKTESTER_H_
#define TNLCUDADEVICECHECKTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/cuda/device-check.h>


#ifdef HAVE_CUDA
__global__ void simpleKernel()
{
   int tIdx = threadIdx. x;
}
#endif


class tnlCudaDeviceCheckTester : public CppUnit :: TestCase
{
   public:
   tnlCudaDeviceCheckTester(){};

   virtual
   ~tnlCudaDeviceCheckTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCudaDeviceCheckTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaDeviceCheckTester >(
                                "deviceTest",
                                &tnlCudaDeviceCheckTester :: deviceTest )
                               );
      return suiteOfTests;
   };

   void deviceTest()
   {
#ifdef HAVE_CUDA
      dim3 blockSize, gridSize;
      blockSize. x = 1;
      gridSize. x = 1;
      simpleKernel<<< gridSize, blockSize >>>();
      if( ! checkCudaDevice )
      {
         cerr << "Test with simple kernel failed. It seems that the CUDA device does not work properly." << endl;
         CPPUNIT_ASSERT( false );
      }
#endif
      CPPUNIT_ASSERT( true );
   };
};


#endif /* TNLCUDADEVICECHECKTESTER_H_ */
