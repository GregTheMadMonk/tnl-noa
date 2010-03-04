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
#include <cppunit/Message.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnlLongVector.h>

#ifdef HAVE_CUDA
void testMultiBlockKernelStarter( const int& number, const int size );
void testMultiBlockKernelStarter( const float& number, const int size );
void testMultiBlockKernelStarter( const double& number, const int size );
void testKernelStarter( const int& number, const int size );
void testKernelStarter( const float& number, const int size );
void testKernelStarter( const double& number, const int size );
int mainTest();
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
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
                               "testAllocation",
                               & tnlLongVectorCUDATester< T > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
                               "testCopying",
                               & tnlLongVectorCUDATester< T > :: testCopying )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
                                     "testAllocationFromNonCUDA",
                                     & tnlLongVectorCUDATester< T > :: testAllocationFromNonCUDA )
                                   );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
                               "testKernel",
                               & tnlLongVectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
							   "testKernel",
                               & tnlLongVectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
      						   "testMultiBlockKernel",
                               & tnlLongVectorCUDATester< T > :: testMultiBlockKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< T > >(
                                                         "testSetValue",
                                     & tnlLongVectorCUDATester< T > :: testSetValue )
                                   );
      return suiteOfTests;
   }

   void testSetValue()
   {
      tnlLongVectorCUDA< T > deviceV( "deviceV", 100 );
      deviceV. setValue( 1.0 );
      tnlLongVector< T > hostV( deviceV );
      hostV. copyFrom( deviceV );
      bool error( false );
      for( int i = 0; i < 100; i ++ )
         if( hostV[ i ] != 1.0 )
            error = true;
      CPPUNIT_ASSERT( ! error );
   }

   void testMultiBlockKernel()
   {
#ifdef HAVE_CUDA
	   for( int size = 100; size <= 10000; size += 100 )
		   for( int i = 0; i < 10; i ++ )
			   :: testMultiBlockKernelStarter( ( T ) i, size );
#else
	   cout << "CUDA is not supported." << endl;
	   CPPUNIT_ASSERT( true );
#endif
   };

   void testKernel()
   {
#ifdef HAVE_CUDA
      for( int i = 0; i < 100; i ++ )
         :: testKernelStarter( ( T ) i, 100 );
#else
      cout << "CUDA is not supported." << endl;
      CPPUNIT_ASSERT( true );
#endif
   };

   void testAllocationFromNonCUDA()
   {
#ifdef HAVE_CUDA

      tnlLongVector< T > hostV( "hostV", 100 );
      for( int i = 0; i < 100; i ++ )
         hostV[ i ] = i;
      tnlLongVectorCUDA< T > deviceV( hostV );
      CPPUNIT_ASSERT( hostV. GetSize() == deviceV. GetSize() );
      deviceV. copyFrom( hostV );
      tnlLongVector< T > hostW( deviceV );
      CPPUNIT_ASSERT( hostW. GetSize() == deviceV. GetSize() );
      hostW. copyFrom( deviceV );
      bool error( false );
      for( int i = 0; i < 100; i ++ )
         if( hostV[ i ] != hostW[ i ] ) error = true;
      CPPUNIT_ASSERT( ! error );
#else
      cout << "CUDA is not supported." << endl;
      CPPUNIT_ASSERT( true );
#endif

   };

   void testCopying()
   {
#ifdef HAVE_CUDA
      tnlLongVector< T > host_vector( "host-vector", 500 );
      tnlLongVectorCUDA< T > device_vector( "device-vector", 500 );
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
 
      cuda_vector_1. setNewSize( 100 );

      CPPUNIT_ASSERT( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_1. GetSize() == 100 );

      tnlLongVectorCUDA< T > cuda_vector_2;
      CPPUNIT_ASSERT( !cuda_vector_2 );
      cuda_vector_2. setNewSize( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_2. GetSize() == 100 );

      tnlLongVectorCUDA< T > cuda_vector_3( "cuda-vector", 100 );
      CPPUNIT_ASSERT( cuda_vector_3. GetSize() == 100 );

      tnlLongVectorCUDA< T >* cuda_vector_4 = new tnlLongVectorCUDA< T >( "cuda-vector-4", 100 );
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
