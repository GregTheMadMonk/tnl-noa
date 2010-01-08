/***************************************************************************
 tnlLongVectorCUDATester.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORCUDATESTER_H_
#define TNLLONGVECTORCUDATESTER_H_

/*
 *
 */
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVector.h>

#ifdef HAVE_CUDA
__global__ void setZeros( float* A );
#endif


template< class T > class tnlLongVectorCUDATester : public CppUnit :: TestCase
{
   public:
   tnlLongVectorCUDATester(){};

   virtual
   ~tnlLongVectorCUDATester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlLongVectorCUDATester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< float > >(
                               "testAllocation",
                               & tnlLongVectorCUDATester< float > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< float > >(
                               "testCopying",
                               & tnlLongVectorCUDATester< float > :: testCopying )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< float > >(
                                     "testKernel",
                                     & tnlLongVectorCUDATester< float > :: testKernel )
                                   );
      return suiteOfTests;
   }

   void testKernel()
   {
#ifdef HAVE_CUDA

#endif
   };

   void testCopying()
   {
#ifdef HAVE_CUDA
      tnlLongVector< T > host_vector( 500 );
      tnlLongVectorCUDA< T > device_vector( 500 );
      for( int i = 0; i < 500; i ++ )
         host_vector[ i ] = ( T ) i;
      device_vector. copyFrom( host_vector );
      host_vector. Zeros();
      host_vector. copyFrom( device_vector );
      int errs( 0 );
      for( int i = 0; i < 500; i ++ )
         if( host_vector[ i ] != i ) errs ++;
      CPPUNIT_ASSERT( ! errs );
#endif
   }

   void testAllocation()
   {
#ifdef HAVE_CUDA
      tnlLongVectorCUDA< T > cuda_vector_1;
      CPPUNIT_ASSERT( !cuda_vector_1 );
 
      cuda_vector_1. SetNewSize( 100 );
      CPPUNIT_ASSERT( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_1. GetSize() == 100 );

      tnlLongVectorCUDA< T > cuda_vector_2;
      CPPUNIT_ASSERT( !cuda_vector_2 );
      cuda_vector_2. SetNewSize( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_2. GetSize() == 100 );

      tnlLongVectorCUDA< T > cuda_vector_3( 100 );
      CPPUNIT_ASSERT( cuda_vector_3. GetSize() == 100 );

      tnlLongVectorCUDA< T >* cuda_vector_4 = new tnlLongVectorCUDA< T >( 100 );
      tnlLongVectorCUDA< T >* cuda_vector_5 = new tnlLongVectorCUDA< T >;
      CPPUNIT_ASSERT( *cuda_vector_4 );
      CPPUNIT_ASSERT( ! *cuda_vector_5 );

      cuda_vector_5 -> SetSharedData( cuda_vector_4 -> Data(),
                                      cuda_vector_4 -> GetSize() );
      CPPUNIT_ASSERT( *cuda_vector_5 );
      /* Shared data are not handled automaticaly.
       * One must be sure that data were not freed sooner
       * then any LongVectors stopped using it.
       */
      delete cuda_vector_5;
      delete cuda_vector_4;
#endif
   }

};

#endif /* TNLLONGVECTORCUDATESTER_H_ */
