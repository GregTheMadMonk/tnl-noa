/***************************************************************************
                          tnlCudaReductionTester.h  -  description
                             -------------------
    begin                : Mar 22, 2013
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


#ifndef TNLCUDAREDUCTIONTESTER_H_
#define TNLCUDAREDUCTIONTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/cuda/device-check.h>
#include <implementation/core/cuda-long-vector-kernels.h>

class tnlCudaReductionTester : public CppUnit :: TestCase
{
   public:
   tnlCudaReductionTester(){};

   virtual
   ~tnlCudaReductionTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCudaReductionTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "shortConstantSequenceTest",
                                &tnlCudaReductionTester :: shortConstantSequenceTest< float > )
                               );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longConstantSequenceTest",
                                &tnlCudaReductionTester :: longConstantSequenceTest< float > )
                               );


      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "linearSequenceTest",
                                &tnlCudaReductionTester :: linearSequenceTest< float > )
                               );


      return suiteOfTests;
   }

   template< typename RealType >
   void setConstantSequence( const int size,
                             const RealType& value,
                             RealType*& hostData,
                             RealType*& deviceData )
   {
      for( int i = 0; i < size; i ++ )
         hostData[ i ] = value;
      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
   }

   template< typename RealType >
   void shortConstantSequenceTest()
   {
      const int shortSequence( 128 );
      const int longSequence( 8192 );
      RealType *hostData, *deviceData;
      allocateMemoryHost( hostData, shortSequence );
      allocateMemoryCuda( deviceData, shortSequence );
      CPPUNIT_ASSERT( checkCudaDevice );

      setConstantSequence( shortSequence, ( RealType ) -1, hostData, deviceData );
      RealType result;

      tnlParallelReductionSum< RealType, int > sumOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( sumOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -shortSequence );

      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( minOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( maxOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absSumOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == shortSequence );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMinOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMaxOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   };

   template< typename RealType >
   void longConstantSequenceTest()
   {
      const int longSequence( 8192 );
      RealType *hostData, *deviceData;
      allocateMemoryHost( hostData, longSequence );
      allocateMemoryCuda( deviceData, longSequence );
      CPPUNIT_ASSERT( checkCudaDevice );

      setConstantSequence( longSequence, ( RealType ) -1, hostData, deviceData );
      RealType result;

      tnlParallelReductionSum< RealType, int > sumOperation;
      CPPUNIT_ASSERT( 
         ( tnlCUDALongVectorReduction( sumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -longSequence );

      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( minOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( maxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absSumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == longSequence );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMinOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMaxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   };

   template< typename RealType >
   void linearSequenceTest()
   {
      const int size( 1024 );
      RealType *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      RealType sum( 0.0 );
      for( int i = 0; i < size; i ++ )
      {
         hostData[ i ] = -i - 1;
         sum += hostData[ i ];
      }
      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      tnlParallelReductionSum< RealType, int > sumOperation;
      RealType result;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( sumOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == sum );
      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( minOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -size );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( maxOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absSumOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == tnlAbs( sum ) );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMinOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( tnlCUDALongVectorReduction( absMaxOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == size );

      freeMemoryHost( hostData );
      freeMemoryCuda( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   };

};


#endif /* TNLCUDAREDUCTIONTESTER_H_ */
