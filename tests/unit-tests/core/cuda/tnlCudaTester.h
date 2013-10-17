/***************************************************************************
                          tnlCudaTester.h  -  description
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

#ifndef TNLCUDATESTER_H_
#define TNLCUDATESTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlCuda.h>


#ifdef HAVE_CUDA
__global__ void simpleKernel()
{
   int tIdx = threadIdx. x;
   tIdx ++;
}
#endif


class tnlCudaTester : public CppUnit :: TestCase
{
   public:
   tnlCudaTester(){};

   virtual
   ~tnlCudaTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCudaTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaTester >(
                                "deviceTest",
                                &tnlCudaTester :: deviceTest )
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

#else
class tnlCudaTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLCUDATESTER_H_ */