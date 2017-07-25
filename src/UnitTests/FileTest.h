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

TEST( FileTest, CloseEmpty )
{
   File file;
   ASSERT_TRUE( file.close() );
}

TEST( FileTest, WriteAndRead )
{
   File file;
   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::write ) );

   int intData( 5 );
   double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;
   ASSERT_TRUE( file.write( &intData ) );
   ASSERT_TRUE( file.write( doubleData, 3 ) );
   ASSERT_TRUE( file.write( &constDoubleData ) );
   ASSERT_TRUE( file.close() );

   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::read ) );
   int newIntData;
   double newDoubleData[ 3 ];
   double newConstDoubleData;
   ASSERT_TRUE( file.read( &newIntData, 1 ) );
   ASSERT_TRUE( file.read( newDoubleData, 3 ) );
   ASSERT_TRUE( file.read( &newConstDoubleData, 1 ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newDoubleData[ i ], doubleData[ i ] );
   EXPECT_EQ( newConstDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
};

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
   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::write ) );

   bool status = file.write< int, Devices::Cuda >( cudaIntData );
   ASSERT_TRUE( status );
   status = file.write< float, Devices::Cuda, int >( cudaFloatData, 3 );
   ASSERT_TRUE( status );
   status = file.write< const double, Devices::Cuda >( cudaConstDoubleData );
   ASSERT_TRUE( status );
   ASSERT_TRUE( file.close() );

   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::read ) );
   int newIntData;
   float newFloatData[ 3 ];
   double newDoubleData;
   int* newCudaIntData;
   float* newCudaFloatData;
   double* newCudaDoubleData;
   cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &newCudaDoubleData, sizeof( double ) );
   status = file.read< int, Devices::Cuda >( newCudaIntData, 1 );
   ASSERT_TRUE( status );
   status = file.read< float, Devices::Cuda, int >( newCudaFloatData, 3 );
   ASSERT_TRUE( status );
   status = file.read< double, Devices::Cuda >( newCudaDoubleData, 1 );
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
