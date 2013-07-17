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

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/cuda/device-check.h>
#include <core/cuda/cuda-reduction.h>

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
                                &tnlCudaReductionTester :: shortConstantSequenceTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longConstantSequenceTest",
                                &tnlCudaReductionTester :: longConstantSequenceTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "linearSequenceTest",
                                &tnlCudaReductionTester :: linearSequenceTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "shortLogicalOperationsTest",
                                &tnlCudaReductionTester :: shortLogicalOperationsTest< int > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longLogicalOperationsTest",
                                &tnlCudaReductionTester :: longLogicalOperationsTest< int > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "shortComparisonTest",
                                &tnlCudaReductionTester :: shortComparisonTest< int > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longComparisonTest",
                                &tnlCudaReductionTester :: longComparisonTest< int > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "shortSdotTest",
                                &tnlCudaReductionTester :: shortSdotTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longSdotTest",
                                &tnlCudaReductionTester :: longSdotTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "shortDiffTest",
                                &tnlCudaReductionTester :: shortDiffTest< double > )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCudaReductionTester >(
                                "longDiffTest",
                                &tnlCudaReductionTester :: longDiffTest< double > )
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
      tnlArrayOperations< tnlHost >::copyMemory< RealType, tnlCuda, RealType, int >( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
   }

   template< typename RealType >
   void shortConstantSequenceTest()
   {
      const int shortSequence( 128 );
      RealType *hostData, *deviceData;
      tnlArrayOperations< tnlHost >::allocateMemory( hostData, shortSequence );
      tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, shortSequence );
      CPPUNIT_ASSERT( checkCudaDevice );

      RealType result;

      setConstantSequence( shortSequence, ( RealType ) -1, hostData, deviceData );
      tnlParallelReductionSum< RealType, int > sumOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( sumOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -shortSequence );

      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( minOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( maxOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absSumOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == shortSequence );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMinOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMaxOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionLpNorm< RealType, int > lpNormOperation;
      lpNormOperation. setPower( 2.0 );
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( lpNormOperation, shortSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == shortSequence );


      tnlArrayOperations< tnlHost >::freeMemory( hostData );
      tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   }

   template< typename RealType >
   void longConstantSequenceTest()
   {
      const int longSequence( 172892 );
      RealType *hostData, *deviceData;
      tnlArrayOperations< tnlHost >::allocateMemory( hostData, longSequence );
      tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, longSequence );
      CPPUNIT_ASSERT( checkCudaDevice );

      RealType result;

      setConstantSequence( longSequence, ( RealType ) -1, hostData, deviceData );
      tnlParallelReductionSum< RealType, int > sumOperation;
      CPPUNIT_ASSERT( 
         ( reductionOnCudaDevice( sumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -longSequence );

      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( minOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( maxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absSumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == longSequence );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMinOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMaxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionLpNorm< RealType, int > lpNormOperation;
      lpNormOperation. setPower( 2.0 );
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( lpNormOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == longSequence );

      setConstantSequence( longSequence, ( RealType ) 2, hostData, deviceData );
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( sumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 * longSequence );

      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( minOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 );

      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( maxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 );

      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absSumOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 * longSequence );

      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMinOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 );

      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMaxOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 2 );

      lpNormOperation. setPower( 2.0 );
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( lpNormOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 4 * longSequence );
      lpNormOperation. setPower( 3.0 );
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( lpNormOperation, longSequence, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 8 * longSequence );


      tnlArrayOperations< tnlHost >::freeMemory( hostData );
      tnlArrayOperations< tnlHost >::freeMemory( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   }

   template< typename RealType >
   void linearSequenceTest()
   {
      const int size( 10245 );
      RealType *hostData, *deviceData;
      tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
      tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      RealType sum( 0.0 );
      for( int i = 0; i < size; i ++ )
      {
         hostData[ i ] = -i - 1;
         sum += hostData[ i ];
      }
      tnlArrayOperations< tnlHost >::copyMemory< RealType, tnlCuda, RealType, int >( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      tnlParallelReductionSum< RealType, int > sumOperation;
      RealType result;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( sumOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == sum );
      tnlParallelReductionMin< RealType, int > minOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( minOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -size );

      tnlParallelReductionMax< RealType, int > maxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( maxOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      tnlParallelReductionAbsSum< RealType, int > absSumOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absSumOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == tnlAbs( sum ) );

      tnlParallelReductionAbsMin< RealType, int > absMinOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMinOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      tnlParallelReductionAbsMax< RealType, int > absMaxOperation;
      CPPUNIT_ASSERT(
         ( reductionOnCudaDevice( absMaxOperation, size, deviceData, ( RealType* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == size );

      tnlArrayOperations< tnlHost >::freeMemory( hostData );
      tnlArrayOperations< tnlCuda >::freeMemory( deviceData );
      CPPUNIT_ASSERT( checkCudaDevice );
   }

   template< typename Type >
   void shortLogicalOperationsTest()
   {
      int size( 125 );
      Type *hostData, *deviceData;
      tnlArrayOperations< tnlHost >::allocateMemory( hostData, size );
      tnlArrayOperations< tnlCuda >::allocateMemory( deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
         hostData[ i ] = 1;

      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      tnlParallelReductionLogicalAnd< Type, int > andOperation;
      tnlParallelReductionLogicalOr< Type, int > orOperation;
      Type result;
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      hostData[ 0 ] = 0;
      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      for( int i = 0; i < size; i ++ )
         hostData[ i ] = 0;

      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
   }

   template< typename Type >
   void longLogicalOperationsTest()
   {
      int size( 7628198 );
      Type *hostData, *deviceData;
      allocateMemoryHost( hostData, size );
      allocateMemoryCuda( deviceData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
         hostData[ i ] = 1;

      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      tnlParallelReductionLogicalAnd< Type, int > andOperation;
      tnlParallelReductionLogicalOr< Type, int > orOperation;
      Type result;
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      hostData[ 0 ] = 0;
      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      for( int i = 0; i < size; i ++ )
         hostData[ i ] = 0;

      copyMemoryHostToCuda( deviceData, hostData, size );
      CPPUNIT_ASSERT( checkCudaDevice );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( andOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( orOperation, size, deviceData, ( Type* ) 0, result ) ) );
      CPPUNIT_ASSERT( result == 0 );
   }

   template< typename Type >
   void shortComparisonTest()
   {
      const int size( 125 );
      Type *hostData1, *hostData2,
           *deviceData1, *deviceData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData1, size );
      allocateMemoryCuda( deviceData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
         hostData1[ i ] = hostData2[ i ] = 1;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      copyMemoryHostToCuda( deviceData2, hostData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      bool result( false );
      tnlParallelReductionEqualities< Type, int > equalityOperation;
      tnlParallelReductionInequalities< Type, int > inequalityOperation;

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == true );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      hostData1[ 0 ] = 0;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      for( int i = 0; i < size; i ++ )
         hostData1[ i ] = 0;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == true );
   }

   template< typename Type >
   void longComparisonTest()
   {
      const int size( 1258976 );
      Type *hostData1, *hostData2,
           *deviceData1, *deviceData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData1, size );
      allocateMemoryCuda( deviceData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
         hostData1[ i ] = hostData2[ i ] = 1;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      copyMemoryHostToCuda( deviceData2, hostData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      bool result( false );
      tnlParallelReductionEqualities< Type, int > equalityOperation;
      tnlParallelReductionInequalities< Type, int > inequalityOperation;

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == true );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      hostData1[ 0 ] = 0;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      for( int i = 0; i < size; i ++ )
         hostData1[ i ] = 0;
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( equalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == false );

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( inequalityOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == true );
   }

   template< typename Type >
   void shortSdotTest()
   {
      const int size( 125 );
      Type *hostData1, *hostData2,
           *deviceData1, *deviceData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData1, size );
      allocateMemoryCuda( deviceData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      hostData1[ 0 ] = 0;
      hostData2[ 0 ] = 1;
      Type sdot( 0.0 );
      for( int i = 1; i < size; i ++ )
      {
         hostData1[ i ] = i;
         hostData2[ i ] = -hostData2[ i - 1 ];
         sdot += hostData1[ i ] * hostData2[ i ];
      }
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      copyMemoryHostToCuda( deviceData2, hostData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      Type result( 0.0 );
      tnlParallelReductionSdot< Type, int > sdotOperation;

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( sdotOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == sdot );
   }


   template< typename Type >
   void longSdotTest()
   {
      const int size( 125789 );
      Type *hostData1, *hostData2,
           *deviceData1, *deviceData2;
      allocateMemoryHost( hostData1, size );
      allocateMemoryHost( hostData2, size );
      allocateMemoryCuda( deviceData1, size );
      allocateMemoryCuda( deviceData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      hostData1[ 0 ] = 0;
      hostData2[ 0 ] = 1;
      Type sdot( 0.0 );
      for( int i = 1; i < size; i ++ )
      {
         hostData1[ i ] = i;
         hostData2[ i ] = -hostData2[ i - 1 ];
         sdot += hostData1[ i ] * hostData2[ i ];
      }
      copyMemoryHostToCuda( deviceData1, hostData1, size );
      copyMemoryHostToCuda( deviceData2, hostData2, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      Type result( 0.0 );
      tnlParallelReductionSdot< Type, int > sdotOperation;

      CPPUNIT_ASSERT(
          ( reductionOnCudaDevice( sdotOperation, size, deviceData1, deviceData2, result ) ) );
      CPPUNIT_ASSERT( result == sdot );
   }

   template< typename Type >
   void shortDiffTest()
   {
      const int size( 123 );
      Type *hostZeros, *hostOnes, *hostLinear,
           *deviceZeros, *deviceOnes, *deviceLinear;
      allocateMemoryHost( hostZeros, size );
      allocateMemoryHost( hostOnes, size );
      allocateMemoryHost( hostLinear, size );
      allocateMemoryCuda( deviceZeros, size );
      allocateMemoryCuda( deviceOnes, size );
      allocateMemoryCuda( deviceLinear, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
      {
         hostZeros[ i ] = 0;
         hostOnes[ i ] = 1;
         hostLinear[ i ] = i;
      }

      copyMemoryHostToCuda( deviceZeros, hostZeros, size );
      copyMemoryHostToCuda( deviceOnes, hostOnes, size );
      copyMemoryHostToCuda( deviceLinear, hostLinear, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      tnlParallelReductionDiffSum< Type, int > diffSumOp;
      tnlParallelReductionDiffMin< Type, int > diffMinOp;
      tnlParallelReductionDiffMax< Type, int > diffMaxOp;
      tnlParallelReductionDiffAbsSum< Type, int > diffAbsSumOp;
      tnlParallelReductionDiffAbsMin< Type, int > diffAbsMinOp;
      tnlParallelReductionDiffAbsMax< Type, int > diffAbsMaxOp;
      tnlParallelReductionDiffLpNorm< Type, int > diffLpNormOp;

      Type result;

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == -size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMinOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMaxOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsSumOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffLpNormOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceLinear, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == size * ( size - 1 ) / 2 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == - size * ( size - 1 ) / 2 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceLinear, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size * ( size - 1 ) / 2 - size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceLinear, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == -size+1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMaxOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == size - 1 );

      freeMemoryHost( hostZeros );
      freeMemoryHost( hostOnes );
      freeMemoryHost( hostLinear );
      freeMemoryCuda( deviceZeros );
      freeMemoryCuda( deviceOnes );
      freeMemoryCuda( deviceLinear );
   }


   template< typename Type >
   void longDiffTest()
   {
      const int size( 12387 );
      Type *hostZeros, *hostOnes, *hostLinear,
           *deviceZeros, *deviceOnes, *deviceLinear;
      allocateMemoryHost( hostZeros, size );
      allocateMemoryHost( hostOnes, size );
      allocateMemoryHost( hostLinear, size );
      allocateMemoryCuda( deviceZeros, size );
      allocateMemoryCuda( deviceOnes, size );
      allocateMemoryCuda( deviceLinear, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      for( int i = 0; i < size; i ++ )
      {
         hostZeros[ i ] = 0;
         hostOnes[ i ] = 1;
         hostLinear[ i ] = i;
      }

      copyMemoryHostToCuda( deviceZeros, hostZeros, size );
      copyMemoryHostToCuda( deviceOnes, hostOnes, size );
      copyMemoryHostToCuda( deviceLinear, hostLinear, size );
      CPPUNIT_ASSERT( checkCudaDevice );

      tnlParallelReductionDiffSum< Type, int > diffSumOp;
      tnlParallelReductionDiffMin< Type, int > diffMinOp;
      tnlParallelReductionDiffMax< Type, int > diffMaxOp;
      tnlParallelReductionDiffAbsSum< Type, int > diffAbsSumOp;
      tnlParallelReductionDiffAbsMin< Type, int > diffAbsMinOp;
      tnlParallelReductionDiffAbsMax< Type, int > diffAbsMaxOp;
      tnlParallelReductionDiffLpNorm< Type, int > diffLpNormOp;

      Type result;

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceOnes, result ) ) );

      CPPUNIT_ASSERT( result == -size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == -1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceOnes, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMinOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMaxOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsSumOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffLpNormOp, size, deviceZeros, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceLinear, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == size * ( size - 1 ) / 2 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == - size * ( size - 1 ) / 2 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffSumOp, size, deviceLinear, deviceOnes, result ) ) );
      CPPUNIT_ASSERT( result == size * ( size - 1 ) / 2 - size );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceLinear, deviceZeros, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMinOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == -size + 1 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffMaxOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == 0 );

      CPPUNIT_ASSERT(
                ( reductionOnCudaDevice( diffAbsMaxOp, size, deviceZeros, deviceLinear, result ) ) );
      CPPUNIT_ASSERT( result == size - 1 );

      freeMemoryHost( hostZeros );
      freeMemoryHost( hostOnes );
      freeMemoryHost( hostLinear );
      freeMemoryCuda( deviceZeros );
      freeMemoryCuda( deviceOnes );
      freeMemoryCuda( deviceLinear );
   }

};

#else
class tnlCudaReductionTester
{};
#endif /* HAVE_CPPUNIT */

#endif /* TNLCUDAREDUCTIONTESTER_H_ */
