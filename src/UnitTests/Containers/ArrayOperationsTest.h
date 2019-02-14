/***************************************************************************
                          ArrayOperationsTest.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST 
#include <TNL/Containers/Algorithms/ArrayOperations.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class ArrayOperationsTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_CASE( ArrayOperationsTest, ValueTypes );

TYPED_TEST( ArrayOperationsTest, allocateMemory_host )
{
   using ValueType = typename TestFixture::ValueType;

   ValueType* data;
   ArrayOperations< Devices::Host >::allocateMemory( data, ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   ArrayOperations< Devices::Host >::freeMemory( data );
}

TYPED_TEST( ArrayOperationsTest, setMemoryElement_host )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *data;
   ArrayOperations< Devices::Host >::allocateMemory( data, size );
   for( int i = 0; i < size; i++ ) {
      ArrayOperations< Devices::Host >::setMemoryElement( data + i, (ValueType) i );
      EXPECT_EQ( data[ i ], i );
      EXPECT_EQ( ArrayOperations< Devices::Host >::getMemoryElement( data + i ), i );
   }
   ArrayOperations< Devices::Host >::freeMemory( data );
}

TYPED_TEST( ArrayOperationsTest, setMemory_host )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *data;
   ArrayOperations< Devices::Host >::allocateMemory( data, size );
   ArrayOperations< Devices::Host >::setMemory( data, (ValueType) 13, size );
   for( int i = 0; i < size; i ++ )
      EXPECT_EQ( data[ i ], 13 );
   ArrayOperations< Devices::Host >::freeMemory( data );
}

TYPED_TEST( ArrayOperationsTest, copyMemory_host )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *data1, *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   ArrayOperations< Devices::Host >::setMemory( data1, (ValueType) 13, size );
   ArrayOperations< Devices::Host >::copyMemory< ValueType, ValueType >( data2, data1, size );
   for( int i = 0; i < size; i ++ )
      EXPECT_EQ( data1[ i ], data2[ i ]);
   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}

TYPED_TEST( ArrayOperationsTest, copyMemoryWithConversion_host )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   int *data1;
   float *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   ArrayOperations< Devices::Host >::setMemory( data1, 13, size );
   ArrayOperations< Devices::Host >::copyMemory< float, int >( data2, data1, size );
   for( int i = 0; i < size; i ++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}

TYPED_TEST( ArrayOperationsTest, compareMemory_host )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *data1, *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   ArrayOperations< Devices::Host >::setMemory( data1, (ValueType) 7, size );
   ArrayOperations< Devices::Host >::setMemory( data2, (ValueType) 0, size );
   EXPECT_FALSE( ( ArrayOperations< Devices::Host >::compareMemory< ValueType, ValueType >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host >::setMemory( data2, (ValueType) 7, size );
   EXPECT_TRUE( ( ArrayOperations< Devices::Host >::compareMemory< ValueType, ValueType >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}

TYPED_TEST( ArrayOperationsTest, compareMemoryWithConversion_host )
{
   const int size = ARRAY_TEST_SIZE;

   int *data1;
   float *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   ArrayOperations< Devices::Host >::setMemory( data1, 7, size );
   ArrayOperations< Devices::Host >::setMemory( data2, (float) 0.0, size );
   EXPECT_FALSE( ( ArrayOperations< Devices::Host >::compareMemory< int, float >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host >::setMemory( data2, (float) 7.0, size );
   EXPECT_TRUE( ( ArrayOperations< Devices::Host >::compareMemory< int, float >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}

TYPED_TEST( ArrayOperationsTest, containsValue_host )
{
   const int size = ARRAY_TEST_SIZE;

   int *data1;
   float *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1[ i ] = i % 10;
      data2[ i ] = ( float ) ( i % 10 );
   }
   for( int i = 0; i < 10; i++ )
   {
      EXPECT_TRUE( ( ArrayOperations< Devices::Host >::containsValue( data1, size, i ) ) );
      EXPECT_TRUE( ( ArrayOperations< Devices::Host >::containsValue( data2, size, ( float ) i ) ) );
   }
   for( int i = 10; i < 20; i++ )
   {
      EXPECT_FALSE( ( ArrayOperations< Devices::Host >::containsValue( data1, size, i ) ) );
      EXPECT_FALSE( ( ArrayOperations< Devices::Host >::containsValue( data2, size, ( float ) i ) ) );
   }
   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}

TYPED_TEST( ArrayOperationsTest, containsOnlyValue_host )
{
   const int size = ARRAY_TEST_SIZE;

   int *data1;
   float *data2;
   ArrayOperations< Devices::Host >::allocateMemory( data1, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2, size );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1[ i ] = i % 10;
      data2[ i ] = ( float ) ( i % 10 );
   }
   for( int i = 0; i < 20; i++ )
   {
      EXPECT_FALSE( ( ArrayOperations< Devices::Host >::containsOnlyValue( data1, size, i ) ) );
      EXPECT_FALSE( ( ArrayOperations< Devices::Host >::containsOnlyValue( data2, size, ( float ) i ) ) );
   }

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1[ i ] = 10;
      data2[ i ] = ( float ) 10;
   }

   EXPECT_TRUE( ( ArrayOperations< Devices::Host >::containsOnlyValue( data1, size, 10 ) ) );
   EXPECT_TRUE( ( ArrayOperations< Devices::Host >::containsOnlyValue( data2, size, ( float ) 10 ) ) );

   ArrayOperations< Devices::Host >::freeMemory( data1 );
   ArrayOperations< Devices::Host >::freeMemory( data2 );
}


#ifdef HAVE_CUDA
TYPED_TEST( ArrayOperationsTest, allocateMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType* data;
   ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   ASSERT_NE( data, nullptr );

   ArrayOperations< Devices::Cuda >::freeMemory( data );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( ArrayOperationsTest, setMemoryElement_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType* data;
   ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );

   for( int i = 0; i < size; i++ )
      ArrayOperations< Devices::Cuda >::setMemoryElement( &data[ i ], (ValueType) i );

   for( int i = 0; i < size; i++ )
   {
      ValueType d;
      ASSERT_EQ( cudaMemcpy( &d, &data[ i ], sizeof( ValueType ), cudaMemcpyDeviceToHost ), cudaSuccess );
      EXPECT_EQ( d, i );
      EXPECT_EQ( ArrayOperations< Devices::Cuda >::getMemoryElement( &data[ i ] ), i );
   }

   ArrayOperations< Devices::Cuda >::freeMemory( data );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( ArrayOperationsTest, setMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *hostData, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, (ValueType) 0, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, (ValueType) 13, size );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< ValueType, ValueType >( hostData, deviceData, size );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( hostData[ i ], 13 );
   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
}

TYPED_TEST( ArrayOperationsTest, copyMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *hostData, *hostData2, *deviceData, *deviceData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, (ValueType) 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< ValueType >( deviceData, hostData, size );
   ArrayOperations< Devices::Cuda >::copyMemory< ValueType, ValueType >( deviceData2, deviceData, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< ValueType, ValueType >( hostData2, deviceData2, size );
   EXPECT_TRUE( ( ArrayOperations< Devices::Host >::compareMemory< ValueType, ValueType >( hostData, hostData2, size) ) );
   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
}

TYPED_TEST( ArrayOperationsTest, copyMemoryWithConversions_cuda )
{
   const int size = ARRAY_TEST_SIZE;

   int *hostData;
   double *hostData2;
   long *deviceData;
   float *deviceData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< long, int >( deviceData, hostData, size );
   ArrayOperations< Devices::Cuda >::copyMemory< float, long >( deviceData2, deviceData, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< double, float >( hostData2, deviceData2, size );
   for( int i = 0; i < size; i ++ )
      EXPECT_EQ( hostData[ i ], hostData2[ i ] );
   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
}

TYPED_TEST( ArrayOperationsTest, compareMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   const int size = ARRAY_TEST_SIZE;

   ValueType *hostData, *deviceData, *deviceData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );

   ArrayOperations< Devices::Host >::setMemory( hostData, (ValueType) 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, (ValueType) 8, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData2, (ValueType) 9, size );
   EXPECT_FALSE(( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< ValueType, ValueType >( hostData, deviceData, size ) ));
   EXPECT_FALSE(( ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< ValueType, ValueType >( deviceData, hostData, size ) ));
   EXPECT_FALSE(( ArrayOperations< Devices::Cuda >::compareMemory< ValueType, ValueType >( deviceData, deviceData2, size ) ));

   ArrayOperations< Devices::Cuda >::setMemory( deviceData, (ValueType) 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData2, (ValueType) 7, size );
   EXPECT_TRUE(( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< ValueType, ValueType >( hostData, deviceData, size ) ));
   EXPECT_TRUE(( ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< ValueType, ValueType >( deviceData, hostData, size ) ));
   EXPECT_TRUE(( ArrayOperations< Devices::Cuda >::compareMemory< ValueType, ValueType >( deviceData, deviceData2, size ) ));

   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
}

TYPED_TEST( ArrayOperationsTest, compareMemoryWithConversions_cuda )
{
   const int size = ARRAY_TEST_SIZE;

   int *hostData;
   float *deviceData;
   double *deviceData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );

   ArrayOperations< Devices::Host >::setMemory( hostData, 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, (float) 8, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData2, (double) 9, size );
   EXPECT_FALSE(( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float >( hostData, deviceData, size ) ));
   EXPECT_FALSE(( ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float, int >( deviceData, hostData, size ) ));
   EXPECT_FALSE(( ArrayOperations< Devices::Cuda >::compareMemory< float, double >( deviceData, deviceData2, size ) ));

   ArrayOperations< Devices::Cuda >::setMemory( deviceData, (float) 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData2, (double) 7, size );
   EXPECT_TRUE(( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float >( hostData, deviceData, size ) ));
   EXPECT_TRUE(( ArrayOperations< Devices::Cuda, Devices::Host >::compareMemory< float, int >( deviceData, hostData, size ) ));
   EXPECT_TRUE(( ArrayOperations< Devices::Cuda >::compareMemory< float, double >( deviceData, deviceData2, size ) ));

   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
}

TYPED_TEST( ArrayOperationsTest, containsValue_cuda )
{
   const int size = ARRAY_TEST_SIZE;

   int *data1_host, *data1_cuda;
   float *data2_host, *data2_cuda;
   ArrayOperations< Devices::Host >::allocateMemory( data1_host, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2_host, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( data1_cuda, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( data2_cuda, size );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1_host[ i ] = i % 10;
      data2_host[ i ] = ( float ) ( i % 10 );
   }

   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data1_cuda, data1_host, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data2_cuda, data2_host, size );

   for( int i = 0; i < 10; i++ )
   {
      EXPECT_TRUE( ( ArrayOperations< Devices::Cuda >::containsValue( data1_cuda, size, i ) ) );
      EXPECT_TRUE( ( ArrayOperations< Devices::Cuda >::containsValue( data2_cuda, size, ( float ) i ) ) );
   }
   for( int i = 10; i < 20; i++ )
   {
      EXPECT_FALSE( ( ArrayOperations< Devices::Cuda >::containsValue( data1_cuda, size, i ) ) );
      EXPECT_FALSE( ( ArrayOperations< Devices::Cuda >::containsValue( data2_cuda, size, ( float ) i ) ) );
   }

   ArrayOperations< Devices::Host >::freeMemory( data1_host );
   ArrayOperations< Devices::Host >::freeMemory( data2_host );
   ArrayOperations< Devices::Cuda >::freeMemory( data1_cuda );
   ArrayOperations< Devices::Cuda >::freeMemory( data2_cuda );
}

TYPED_TEST( ArrayOperationsTest, containsOnlyValue_cuda )
{
   const int size = ARRAY_TEST_SIZE;

   int *data1_host, *data1_cuda;
   float *data2_host, *data2_cuda;
   ArrayOperations< Devices::Host >::allocateMemory( data1_host, size );
   ArrayOperations< Devices::Host >::allocateMemory( data2_host, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( data1_cuda, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( data2_cuda, size );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1_host[ i ] = i % 10;
      data2_host[ i ] = ( float ) ( i % 10 );
   }
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data1_cuda, data1_host, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data2_cuda, data2_host, size );

   for( int i = 0; i < 20; i++ )
   {
      EXPECT_FALSE( ( ArrayOperations< Devices::Cuda >::containsOnlyValue( data1_cuda, size, i ) ) );
      EXPECT_FALSE( ( ArrayOperations< Devices::Cuda >::containsOnlyValue( data2_cuda, size, ( float ) i ) ) );
   }

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      data1_host[ i ] = 10;
      data2_host[ i ] = ( float ) 10;
   }
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data1_cuda, data1_host, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory( data2_cuda, data2_host, size );

   EXPECT_TRUE( ( ArrayOperations< Devices::Cuda >::containsOnlyValue( data1_cuda, size, 10 ) ) );
   EXPECT_TRUE( ( ArrayOperations< Devices::Cuda >::containsOnlyValue( data2_cuda, size, ( float ) 10 ) ) );

   ArrayOperations< Devices::Host >::freeMemory( data1_host );
   ArrayOperations< Devices::Host >::freeMemory( data2_host );
   ArrayOperations< Devices::Cuda >::freeMemory( data1_cuda );
   ArrayOperations< Devices::Cuda >::freeMemory( data2_cuda );
}
#endif // HAVE_CUDA
#endif // HAVE_GTEST


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
