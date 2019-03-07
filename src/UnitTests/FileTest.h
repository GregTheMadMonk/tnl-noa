/***************************************************************************
                          FileTest.h  -  description
                             -------------------
    begin                : Oct 24, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/File.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

using namespace TNL;

TEST( FileTest, OpenInvalid )
{
   File file;
   EXPECT_THROW( file.open( "invalid-file.tnl", File::Mode::In ), std::ios_base::failure );
}

TEST( FileTest, WriteAndRead )
{
   File file;
   file.open( String( "test-file.tnl" ), File::Mode::Out );

   int intData( 5 );
   double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;
   ASSERT_TRUE( file.save( &intData ) );
   ASSERT_TRUE( file.save( doubleData, 3 ) );
   ASSERT_TRUE( file.save( &constDoubleData ) );
   file.close();

   file.open( String( "test-file.tnl" ), File::Mode::In );
   int newIntData;
   double newDoubleData[ 3 ];
   double newConstDoubleData;
   ASSERT_TRUE( file.load( &newIntData, 1 ) );
   ASSERT_TRUE( file.load( newDoubleData, 3 ) );
   ASSERT_TRUE( file.load( &newConstDoubleData, 1 ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newDoubleData[ i ], doubleData[ i ] );
   EXPECT_EQ( newConstDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
};

TEST( FileTest, WriteAndReadWithConversion )
{
   double doubleData[ 3 ] = {  3.1415926535897932384626433,
                               2.7182818284590452353602874,
                               1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];
   File file;
   file.open( "test-file.tnl", File::Mode::Out | File::Mode::Truncate );
   file.save< double, float, Devices::Host >( doubleData, 3 );
   file.close();

   file.open( "test-file.tnl", File::Mode::In );
   file.load< float, float, Devices::Host >( floatData, 3 );
   file.close();

   file.open( "test-file.tnl", File::Mode::In );
   file.load< int, float, Devices::Host >( intData, 3 );
   file.close();

   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

#ifdef HAVE_CUDA
TEST( FileTest, WriteAndReadCUDA )
{
   int intData( 5 );
   float floatData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   cudaMalloc( ( void** ) &cudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &cudaConstDoubleData, sizeof( double ) );
   cudaMemcpy( cudaIntData,
               &intData,
               sizeof( int ),
               cudaMemcpyHostToDevice );
   cudaMemcpy( cudaFloatData,
               floatData,
               3 * sizeof( float ),
               cudaMemcpyHostToDevice );
   cudaMemcpy( (void*) cudaConstDoubleData,
               &constDoubleData,
               sizeof( double ),
               cudaMemcpyHostToDevice );

   File file;
   file.open( String( "test-file.tnl" ), File::Mode::Out );

   bool status = file.save< int, int, Devices::Cuda >( cudaIntData );
   ASSERT_TRUE( status );
   status = file.save< float, float, Devices::Cuda >( cudaFloatData, 3 );
   ASSERT_TRUE( status );
   status = file.save< const double, double, Devices::Cuda >( cudaConstDoubleData );
   ASSERT_TRUE( status );
   file.close();

   file.open( String( "test-file.tnl" ), File::Mode::In );
   int newIntData;
   float newFloatData[ 3 ];
   double newDoubleData;
   int* newCudaIntData;
   float* newCudaFloatData;
   double* newCudaDoubleData;
   cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &newCudaDoubleData, sizeof( double ) );
   status = file.load< int, int, Devices::Cuda >( newCudaIntData, 1 );
   ASSERT_TRUE( status );
   status = file.load< float, float, Devices::Cuda >( newCudaFloatData, 3 );
   ASSERT_TRUE( status );
   status = file.load< double, double, Devices::Cuda >( newCudaDoubleData, 1 );
   ASSERT_TRUE( status );
   cudaMemcpy( &newIntData,
               newCudaIntData,
               sizeof( int ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( newFloatData,
               newCudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( &newDoubleData,
               newCudaDoubleData,
               sizeof( double ),
               cudaMemcpyDeviceToHost );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newFloatData[ i ], floatData[ i ] );
   EXPECT_EQ( newDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
};

TEST( FileTest, WriteAndReadCUDAWithConversion )
{
   const double constDoubleData[ 3 ] = {  3.1415926535897932384626433,
                                          2.7182818284590452353602874,
                                          1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   cudaMalloc( ( void** ) &cudaIntData, 3 * sizeof( int ) );
   cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &cudaConstDoubleData, 3 * sizeof( double ) );
   cudaMemcpy( (void*) cudaConstDoubleData,
               &constDoubleData,
               3 * sizeof( double ),
               cudaMemcpyHostToDevice );

   File file;
   file.open( String( "cuda-test-file.tnl" ), File::Mode::Out | File::Mode::Truncate );
   file.save< double, float, Devices::Cuda >( cudaConstDoubleData, 3 );
   file.close();

   file.open( String( "cuda-test-file.tnl" ), File::Mode::In );
   file.load< float, float, Devices::Cuda >( cudaFloatData, 3 );
   file.close();

   file.open( String( "cuda-test-file.tnl" ), File::Mode::In );
   file.load< int, float, Devices::Cuda >( cudaIntData, 3 );
   file.close();

   cudaMemcpy( floatData,
               cudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( &intData,
               cudaIntData,
               3* sizeof( int ),
               cudaMemcpyDeviceToHost );


   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( "cuda-test-file.tnl" ), 0 );
};

#endif
#endif

#include "GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
