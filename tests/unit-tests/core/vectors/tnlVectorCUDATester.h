/***************************************************************************
 VectorCUDATester.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
 email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
#include <TNL/Vectors/VectorCUDA.h>
#include <TNL/Vectors/VectorHost.h>

#ifdef HAVE_CUDA
//int mainTest();

using namespace TNL;

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


template< class T > class VectorCUDATester : public CppUnit :: TestCase
{
   public:
   VectorCUDATester(){};

   virtual
   ~VectorCUDATester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "VectorCUDATester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testAllocation",
                               & VectorCUDATester< T > :: testAllocation )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testCopying",
                               & VectorCUDATester< T > :: testCopying )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testAllocationFromNonCUDA",
                               & VectorCUDATester< T > :: testAllocationFromNonCUDA )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testKernel",
                               & VectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
							          "testKernel",
                               & VectorCUDATester< T > :: testKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testMultiBlockKernel",
                               & VectorCUDATester< T > :: testMultiBlockKernel )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testSetValue",
                               & VectorCUDATester< T > :: testSetValue )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testSetElement",
                               & VectorCUDATester< T > :: testSetElement )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testComparison",
                               & VectorCUDATester< T > :: testComparison )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testParallelReduciontMethods",
                               & VectorCUDATester< T > :: testParallelReduciontMethods )
                               );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< VectorCUDATester< T > >(
                               "testBlasFunctions",
                               & VectorCUDATester< T > :: testBlasFunctions )
                               );

      return suiteOfTests;
   }

   void testBlasFunctions()
   {
      Vector< T, Devices::Cuda > u( "VectorTester :: u" );
      Vector< T, Devices::Cuda > v( "VectorTester :: v" );
      Vector< T, Devices::Cuda > w( "VectorTester :: w" );
      u. setSize( 16 );
      v. setSize( u. getSize() );
      w. setSize( u. getSize() );
      u. setValue( 2 );
      v. setValue( 3 );
      w. setValue( 8 );
      CPPUNIT_ASSERT( tnlSDOT( u, v ) == 6 * u. getSize() );

      tnlalphaXPlusY( ( T ) 2, v, u );
      CPPUNIT_ASSERT( u == w );
   }

   void testParallelReduciontMethods()
   {
      Vector< T, Devices::Cuda > u( "VectorCUDATester :: u" );
      Vector< T, Devices::Cuda > w( "VectorCUDATester :: w" );

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
      Vector< T, Devices::Host > v( "VectorCUDATester :: v" );
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
      CPPUNIT_ASSERT( abs( tnlLpNorm( u, ( T ) 1 ) - tnlLpNorm( v, ( T ) 1 ) ) < 0.01 );

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
      CPPUNIT_ASSERT( abs( tnlLpNorm( u, ( T ) 1 ) - tnlLpNorm( v, ( T ) 1 ) ) < 0.01 );

      w. setValue( 2 );
      u. setValue( 2 );
      CPPUNIT_ASSERT( tnlSDOT( u, w ) == 4 * u. getSize() );



   };


   void testComparison()
   {
	   //cerr << "testComparison" << std::endl;
      Vector< T, Devices::Cuda > deviceV( "deviceV", 100 );
      Vector< T, Devices::Cuda > deviceU( "deviceU", 100 );
      deviceV. setValue( 1.0 );
      deviceU. setValue( 1.0 );
      Vector< T, Devices::Host > hostV( "hostV", deviceV );

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
      Vector< T, Devices::Cuda > deviceV( "deviceV", 100 );
      Vector< T, Devices::Host > hostV( "hostV", 100 );
      for( int i = 0; i < 100; i ++ )
      {
         deviceV. setElement( i, i );
         hostV. setElement( i, i );
      }
      Vector< T, Devices::Host > hostU( "hostU", 100 );
      hostU = deviceV;
      CPPUNIT_ASSERT( hostU == hostV );
   }

   void testSetValue()
   {
	   //cerr << "testSetValue" << std::endl;
      Vector< T, Devices::Cuda > deviceV( "deviceV", 100 );
      deviceV. setValue( 1 );
      Vector< T, Devices::Host > hostV( "hostV", deviceV );
      hostV = deviceV;
      bool error( false );
      for( int i = 0; i < 100; i ++ )
         if( hostV[ i ] != 1.0 )
            error = true;
      CPPUNIT_ASSERT( ! error );
   }

   void testMultiBlockKernel()
   {
	   //cerr << "testMultiBlockKernel" << std::endl;
#ifdef HAVE_CUDA
	   const T number = 1.0;
	   for( int size = 100; size <= 10000; size += 100 )
		   for( int i = 0; i < 10; i ++ )
		   {
		      Vector< T, Devices::Cuda > device_vector( "device-vector", size );
		      Vector< T, Devices::Host > host_vector( "host-vector", size );
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
	  std::cout << "CUDA is not supported." << std::endl;
	   CPPUNIT_ASSERT( true );
#endif
   };

   void testKernel()
   {
	   //cerr << "testKernel" << std::endl;
#ifdef HAVE_CUDA
	   const int size = 100;
	   const T number = 1.0;
      for( int i = 0; i < 100; i ++ )
      {
         Vector< T, Devices::Cuda > device_vector( "device-vector", size );
         Vector< T, Devices::Host > host_vector( "host-vector", size );
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
     std::cout << "CUDA is not supported." << std::endl;
      CPPUNIT_ASSERT( true );
#endif
   };

   void testAllocationFromNonCUDA()
   {
	   //cerr << "testAllocationFromNonCUDA" << std::endl;
#ifdef HAVE_CUDA

      Vector< T > hostV( "hostV", 100 );
      for( int i = 0; i < 100; i ++ )
         hostV[ i ] = i;
      Vector< T, Devices::Cuda > deviceV( "deviceV", hostV );
      CPPUNIT_ASSERT( hostV. getSize() == deviceV. getSize() );
      deviceV = hostV;
      Vector< T, Devices::Host > hostW( "hostW", deviceV );
      CPPUNIT_ASSERT( hostW. getSize() == deviceV. getSize() );
      hostW = deviceV;
      bool error( false );
      for( int i = 0; i < 100; i ++ )
         if( hostV[ i ] != hostW[ i ] ) error = true;
      CPPUNIT_ASSERT( ! error );
#else
     std::cout << "CUDA is not supported." << std::endl;
      CPPUNIT_ASSERT( true );
#endif

   };

   void testCopying()
   {
	   //cerr << "testCopying - Long Vector" << std::endl;
#ifdef HAVE_CUDA
      Vector< T, Devices::Host > host_vector( "host-vector", 500 );
      Vector< T, Devices::Cuda > device_vector( "device-vector", 500 );
      for( int i = 0; i < 500; i ++ )
         host_vector[ i ] = ( T ) i;
      device_vector = host_vector;
      host_vector. setValue( 0 );
      host_vector = device_vector;
      int errs( 0 );
      for( int i = 0; i < 500; i ++ )
         if( host_vector[ i ] != i ) errs ++;
      CPPUNIT_ASSERT( ! errs );
      Vector< T, Devices::Cuda > device_vector2( "device-vector2", 500 );
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
	   //cerr << "testAllocation - Long Vector" << std::endl;
      Vector< T, Devices::Cuda > cuda_vector_1( "VectorCUDATester:cuda-vector-1" );
      CPPUNIT_ASSERT( !cuda_vector_1 );
 
      cuda_vector_1. setSize( 100 );

      CPPUNIT_ASSERT( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_1. getSize() == 100 );

      Vector< T, Devices::Cuda > cuda_vector_2( "VectorCUDATester:cuda-vector-2" );
      CPPUNIT_ASSERT( !cuda_vector_2 );
      cuda_vector_2. setSize( cuda_vector_1 );
      CPPUNIT_ASSERT( cuda_vector_2. getSize() == 100 );

      Vector< T, Devices::Cuda > cuda_vector_3( "VectorCUDATester:cuda-vector-3", 100 );
      CPPUNIT_ASSERT( cuda_vector_3. getSize() == 100 );

      Vector< T, Devices::Cuda >* cuda_vector_4 = new Vector< T, Devices::Cuda >( "VectorCUDATester:cuda-vector-4", 100 );
      Vector< T, Devices::Cuda >* cuda_vector_5 = new Vector< T, Devices::Cuda >( "VectorCUDATester:cuda-vector-5");
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
