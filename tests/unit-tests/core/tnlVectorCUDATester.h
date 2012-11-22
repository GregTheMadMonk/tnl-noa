/***************************************************************************
 tnlVectorCUDATester.h  -  description
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
#include <core/tnlVectorCUDA.h>
#include <core/tnlVectorHost.h>

#ifdef HAVE_CUDA
//int mainTest();

template< class T >
__global__ void setMultiBlockNumber( const T c, T* A, const int size )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size ) A[ i ] = c;
};

template< class T >
__global__ void setNumber( const T c, T* A, const int size )
{
   int i = threadIdx. x;
   if( i < size )
      A[ i ] = c;
};
#endif


template< class T > class tnlVectorCUDATester : public CppUnit :: TestCase
{
   public:
   tnlVectorCUDATester(){};

   virtual
   ~tnlVectorCUDATester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlVectorCUDATester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testAllocation",
                               & tnlVectorCUDATester< T > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testCopying",
                               & tnlVectorCUDATester< T > :: testCopying )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testAllocationFromNonCUDA",
                               & tnlVectorCUDATester< T > :: testAllocationFromNonCUDA )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testKernel",
                               & tnlVectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
							          "testKernel",
                               & tnlVectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testMultiBlockKernel",
                               & tnlVectorCUDATester< T > :: testMultiBlockKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testSetValue",
                               & tnlVectorCUDATester< T > :: testSetValue )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testSetElement",
                               & tnlVectorCUDATester< T > :: testSetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testComparison",
                               & tnlVectorCUDATester< T > :: testComparison )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testParallelReduciontMethods",
                               & tnlVectorCUDATester< T > :: testParallelReduciontMethods )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlVectorCUDATester< T > >(
                               "testBlasFunctions",
                               & tnlVectorCUDATester< T > :: testBlasFunctions )
                               );

      return suiteOfTests;
   }

   void testBlasFunctions()
   {
      tnlVector< T, tnlCuda > u( "tnlVectorTester :: u" );
      tnlVector< T, tnlCuda > v( "tnlVectorTester :: v" );
      tnlVector< T, tnlCuda > w( "tnlVectorTester :: w" );
      u. setSize( 16 );
      v. setSize( u. getSize() );
      w. setSize( u. getSize() );
      u. setValue( 2 );
      v. setValue( 3 );
      w. setValue( 8 );
      CPPUNIT_ASSERT( tnlSDOT( u, v ) == 6 * u. getSize() );

      tnlSAXPY( ( T ) 2, v, u );
      CPPUNIT_ASSERT( u == w );
   }

   void testParallelReduciontMethods()
   {
      tnlVector< T, tnlCuda > u( "tnlVectorCUDATester :: u" );
      tnlVector< T, tnlCuda > w( "tnlVectorCUDATester :: w" );

      /****
       * We first test with smaller size of the vector. The reduction is done on
       * CPU in this case.
       */

      u. setSize( 10 );
      u. setValue( 0 );
      w. setSize( u. getSize() );
      w. setValue( 1 );
      CPPUNIT_ASSERT( tnlMax( u ) == 0 );
      CPPUNIT_ASSERT( tnlMin( u ) == 0 );
      CPPUNIT_ASSERT( tnlAbsMax( u ) == 0 );
      CPPUNIT_ASSERT( tnlAbsMin( u ) == 0 );
      CPPUNIT_ASSERT( tnlLpNorm( u, ( T ) 1 ) == 0 );
      CPPUNIT_ASSERT( tnlSum( u ) == 0 );
      CPPUNIT_ASSERT( tnlSDOT( u, w ) == 0 );

      for( int i = 0; i < 10; i ++ )
         u. setElement( i, -i );

      CPPUNIT_ASSERT( tnlMax( u ) == 0 );
      CPPUNIT_ASSERT( tnlMin( u ) == - 9 );
      CPPUNIT_ASSERT( tnlSum( u ) == -45 );
      CPPUNIT_ASSERT( tnlAbsMax( u ) == 9 );
      CPPUNIT_ASSERT( tnlAbsMin( u ) == 0 );
      CPPUNIT_ASSERT( tnlLpNorm( u, ( T ) 1 ) == 45 );
      CPPUNIT_ASSERT( tnlSDOT( u, w ) == -45 );

      /****
       * Now we will test with larger vector
       */
      tnlVector< T, tnlHost > v( "tnlVectorCUDATester :: v" );
      u. setSize( 65536 );
      v. setSize( u. getSize() );

      for( int i = 0; i < v. getSize(); i ++ )
         v. setElement( i, i );
      u = v;

      CPPUNIT_ASSERT( tnlMax( u ) == tnlMax( v ) );
      CPPUNIT_ASSERT( tnlMin( u ) == tnlMin( v ) );
      CPPUNIT_ASSERT( fabs( tnlSum( u ) - tnlSum( v ) ) < 0.0001 * 1.0e9 );
      CPPUNIT_ASSERT( tnlAbsMax( u ) == tnlAbsMax( v ) );
      CPPUNIT_ASSERT( tnlAbsMin( u ) == tnlAbsMin( v ) );

      u. setValue( ( T ) -1 );
      v. setValue( ( T ) -1 );
      CPPUNIT_ASSERT( tnlAbs( tnlLpNorm( u, ( T ) 1 ) - tnlLpNorm( v, ( T ) 1 ) ) < 0.01 );

      w. setSize( u. getSize() );
      w. setValue( 2 );
      u. setValue( 2 );
      CPPUNIT_ASSERT( tnlSDOT( u, w ) == 4 * u. getSize() );

      /****
       * And now with even longer vector to get more then one reduction step.
       */

      u. setSize( 1 << 22 );
      v. setSize( u. getSize() );
      w. setSize( u. getSize() );
      u. setValue( 1.0 );
      v. setValue( 1.0 );
      CPPUNIT_ASSERT( tnlMax( u ) == tnlMax( v ) );
      CPPUNIT_ASSERT( tnlMin( u ) == tnlMin( v ) );
      CPPUNIT_ASSERT( fabs( tnlSum( u ) - tnlSum( v ) ) < 0.0001 * 1.0e9 );
      CPPUNIT_ASSERT( tnlAbsMax( u ) == tnlAbsMax( v ) );
      CPPUNIT_ASSERT( tnlAbsMin( u ) == tnlAbsMin( v ) );

      u. setValue( ( T ) -1 );
      v. setValue( ( T ) -1 );
      CPPUNIT_ASSERT( tnlAbs( tnlLpNorm( u, ( T ) 1 ) - tnlLpNorm( v, ( T ) 1 ) ) < 0.01 );

      w. setValue( 2 );
      u. setValue( 2 );
      CPPUNIT_ASSERT( tnlSDOT( u, w ) == 4 * u. getSize() );



   };


   void testComparison()
   {
	   //cerr << "testComparison" << endl;
      tnlVector< T, tnlCuda > deviceV( "deviceV", 100 );
      tnlVector< T, tnlCuda > deviceU( "deviceU", 100 );
      deviceV. setValue( 1.0 );
      deviceU. setValue( 1.0 );
      tnlVector< T, tnlHost > hostV( "hostV", deviceV );

      hostV. setValue( 1.0 );
      CPPUNIT_ASSERT( deviceV == hostV );
      CPPUNIT_ASSERT( hostV == deviceV );
      CPPUNIT_ASSERT( deviceV == deviceU );

      deviceV. setValue( 0.0 );
      CPPUNIT_ASSERT( deviceV != hostV );
      CPPUNIT_ASSERT( hostV != deviceV );
      CPPUNIT_ASSERT( deviceV != deviceU );

   }

   void testSetElement()
   {
      tnlVector< T, tnlCuda > deviceV( "deviceV", 100 );
      tnlVector< T, tnlHost > hostV( "hostV", 100 );
      for( int i = 0; i < 100; i ++ )
      {
         deviceV. setElement( i, i );
         hostV. setElement( i, i );
      }
      tnlVector< T, tnlHost > hostU( "hostU", 100 );
      hostU = deviceV;
      CPPUNIT_ASSERT( hostU == hostV );
   }

   void testSetValue()
   {
	   //cerr << "testSetValue" << endl;
      tnlVector< T, tnlCuda > deviceV( "deviceV", 100 );
      deviceV. setValue( 1 );
      tnlVector< T, tnlHost > hostV( "hostV", deviceV );
      hostV = deviceV;
      bool error( false );
      for( int i = 0; i < 100; i ++ )
         if( hostV[ i ] != 1.0 )
            error = true;
      CPPUNIT_ASSERT( ! error );
   }

   void testMultiBlockKernel()
   {
	   //cerr << "testMultiBlockKernel" << endl;
#ifdef HAVE_CUDA
	   const T number = 1.0;
	   for( int size = 100; size <= 10000; size += 100 )
		   for( int i = 0; i < 10; i ++ )
		   {
		      tnlVector< T, tnlCuda > device_vector( "device-vector", size );
		      tnlVector< T, tnlHost > host_vector( "host-vector", size );
		      T* data = device_vector. getData();

		      const int block_size = 512;
		      const int grid_size = size / 512 + 1;

		      setMultiBlockNumber<<< grid_size, block_size >>>( number, data, size );
		      host_vector = device_vector;

		      int errors( 0 );
		      for( int i = 0; i < size; i ++ )
		      {
		         //cout << host_vector[ i ] << "-";
		         if( host_vector[ i ] != number ) errors ++;
		      }
		      CPPUNIT_ASSERT( ! errors );
		   }
#else
	   cout << "CUDA is not supported." << endl;
	   CPPUNIT_ASSERT( true );
#endif
   };

   void testKernel()
   {
	   //cerr << "testKernel" << endl;
#ifdef HAVE_CUDA
	   const int size = 100;
	   const T number = 1.0;
      for( int i = 0; i < 100; i ++ )
      {
         tnlVector< T, tnlCuda > device_vector( "device-vector", size );
         tnlVector< T, tnlHost > host_vector( "host-vector", size );
         T* data = device_vector. getData();
         setNumber<<< 1, size >>>( number, data, size );
         host_vector = device_vector;

         int errors( 0 );
         for( int i = 0; i < size; i ++ )
         {
            //cout << host_vector[ i ] << "-";
            if( host_vector[ i ] != number ) errors ++;
         }
         CPPUNIT_ASSERT( ! errors );
      }
#else
      cout << "CUDA is not supported." << endl;
      CPPUNIT_ASSERT( true );
#endif
   };

   void testAllocationFromNonCUDA()
   {
	   //cerr << "testAllocationFromNonCUDA" << endl;
#ifdef HAVE_CUDA

      tnlVector< T > hostV( "hostV", 100 );
      for( int i = 0; i < 100; i ++ )
         hostV[ i ] = i;
      tnlVector< T, tnlCuda > deviceV( "deviceV", hostV );
      CPPUNIT_ASSERT( hostV. getSize() == deviceV. getSize() );
      deviceV = hostV;
      tnlVector< T, tnlHost > hostW( "hostW", deviceV );
      CPPUNIT_ASSERT( hostW. getSize() == deviceV. getSize() );
      hostW = deviceV;
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
	   //cerr << "testCopying - Long Vector" << endl;
#ifdef HAVE_CUDA
      tnlVector< T, tnlHost > host_vector( "host-vector", 500 );
      tnlVector< T, tnlCuda > device_vector( "device-vector", 500 );
      for( int i = 0; i < 500; i ++ )
         host_vector[ i ] = ( T ) i;
      device_vector = host_vector;
      host_vector. setValue( 0 );
      host_vector = device_vector;
      int errs( 0 );
      for( int i = 0; i < 500; i ++ )
         if( host_vector[ i ] != i ) errs ++;
      CPPUNIT_ASSERT( ! errs );
      tnlVector< T, tnlCuda > device_vector2( "device-vector2", 500 );
      device_vector2 = device_vector;
      host_vector. setValue( 0 );
      host_vector = device_vector2;
      for( int i = 0; i < 500; i ++ )
         if( host_vector[ i ] != i ) errs ++;
#endif
   }

   void testAllocation()
   {
#ifdef HAVE_CUDA
	   //cerr << "testAllocation - Long Vector" << endl;
      tnlVector< T, tnlCuda > cuda_vector_1( "tnlVectorCUDATester:cuda-vector-1" );
      CPPUNIT_ASSERT( !cuda_vector_1 );
 
      cuda_vector_1. setSize( 100 );

      CPPUNIT_ASSERT( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_1. getSize() == 100 );

      tnlVector< T, tnlCuda > cuda_vector_2( "tnlVectorCUDATester:cuda-vector-2" );
      CPPUNIT_ASSERT( !cuda_vector_2 );
      cuda_vector_2. setSize( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_2. getSize() == 100 );

      tnlVector< T, tnlCuda > cuda_vector_3( "tnlVectorCUDATester:cuda-vector-3", 100 );
      CPPUNIT_ASSERT( cuda_vector_3. getSize() == 100 );

      tnlVector< T, tnlCuda >* cuda_vector_4 = new tnlVector< T, tnlCuda >( "tnlVectorCUDATester:cuda-vector-4", 100 );
      tnlVector< T, tnlCuda >* cuda_vector_5 = new tnlVector< T, tnlCuda >( "tnlVectorCUDATester:cuda-vector-5");
      CPPUNIT_ASSERT( *cuda_vector_4 );
      CPPUNIT_ASSERT( ! *cuda_vector_5 );

      /*cuda_vector_5 -> bind( cuda_vector_4 -> getData(),
                                      cuda_vector_4 -> getSize() );
      CPPUNIT_ASSERT( *cuda_vector_5 );*/
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
