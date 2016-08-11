/***************************************************************************
                          Devices::CudaTester.h  -  description
                             -------------------
    begin                : Mar 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlCudaTESTER_H_
#define tnlCudaTESTER_H_

#include <TNL/tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Devices/Cuda.h>


#ifdef HAVE_CUDA
__global__ void simpleKernel()
{
   int tIdx = threadIdx. x;
   tIdx ++;
}
#endif

using namespace TNL;

class Devices::CudaTester : public CppUnit :: TestCase
{
   public:
   Devices::CudaTester(){};

   virtual
   ~Devices::CudaTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "Devices::CudaTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< Devices::CudaTester >(
                                "deviceTest",
                                &Devices::CudaTester :: deviceTest )
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
         std::cerr << "Test with simple kernel failed. It seems that the CUDA device does not work properly." << std::endl;
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

#endif /* Devices::CudaTESTER_H_ */
