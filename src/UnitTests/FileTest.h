/***************************************************************************
                          tnlFileTester.h  -  description
                             -------------------
    begin                : Oct 24, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/File.h>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST
TEST( FileTest, WriteAndRead )
{
   File file;
   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::write ) );

   int intData( 5 );
   double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
   ASSERT_TRUE( file.write( &intData ) );
   ASSERT_TRUE( file.write( doubleData, 3 ) );
   ASSERT_TRUE( file.close() );

   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::read ) );
   int newIntData;
   double newDoubleData[ 3 ];
   ASSERT_TRUE( file.read( &newIntData, 1 ) );
   ASSERT_TRUE( file.read( newDoubleData, 3 ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newDoubleData[ i ], doubleData[ i ] );
};

#ifdef HAVE_CUDA
TEST( FileTest, WriteAndReadCUDA )
{
   int intData( 5 );
   float floatData[ 3 ] = { 1.0, 2.0, 3.0 };

   int* cudaIntData;
   float* cudaFloatData;
   cudaMalloc( ( void** ) &cudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
   cudaMemcpy( cudaIntData,
               &intData,
               sizeof( int ),
               cudaMemcpyHostToDevice );
   cudaMemcpy( cudaFloatData,
               floatData,
               3 * sizeof( float ),
               cudaMemcpyHostToDevice );

   File file;
   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::write ) );

   bool status = file.write< int, Devices::Cuda >( cudaIntData );
   ASSERT_TRUE( status );
   status = file.write< float, Devices::Cuda, int >( cudaFloatData, 3 );
   ASSERT_TRUE( status );
   ASSERT_TRUE( file.close() );

   ASSERT_TRUE( file.open( String( "test-file.tnl" ), IOMode::read ) );
   int newIntData;
   float newFloatData[ 3 ];
   int* newCudaIntData;
   float* newCudaFloatData;
   cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
   status = file.read< int, Devices::Cuda >( newCudaIntData, 1 );
   ASSERT_TRUE( status );
   status = file.read< float, Devices::Cuda, int >( newCudaFloatData, 3 );
   ASSERT_TRUE( status );
   cudaMemcpy( &newIntData,
               newCudaIntData,
               sizeof( int ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( newFloatData,
               newCudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newFloatData[ i ], floatData[ i ] );
};
#endif
#endif

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}
