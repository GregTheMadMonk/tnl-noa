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
#include "gtest/gtest.h"
#endif

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST
TEST( FileTest, WriteAndRead )
{
   File file;
   if( ! file. open( String( "test-file.tnl" ), tnlWriteMode ) )
   {
      std::cerr << "Unable to create file test-file.tnl for the testing." << std::endl;
      return;
   }
   int intData( 5 );
#ifdef HAVE_NOT_CXX11
   file. write< int, Devices::Host >( &intData );
#else
   file. write( &intData );
#endif
   double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
#ifdef HAVE_NOT_CXX11
   file. write< double, Devices::Host >( doubleData, 3 );
#else
   file. write( doubleData, 3 );
#endif
   if( ! file. close() )
   {
      std::cerr << "Unable to close the file test-file.tnl" << std::endl;
      return;
   }

   if( ! file. open( String( "test-file.tnl" ), tnlReadMode ) )
   {
      std::cerr << "Unable to open the file test-file.tnl for the testing." << std::endl;
      return;
   }
   int newIntData;
   double newDoubleData[ 3 ];
#ifdef HAVE_NOT_CXX11
   file. read< int, Devices::Host >( &newIntData );
   file. read< double, Devices::Host >( newDoubleData, 3 );
#else
   file. read( &newIntData, 1 );
   file. read( newDoubleData, 3 );
#endif

   ASSERT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      ASSERT_EQ( newDoubleData[ i ], doubleData[ i ] );
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
   if( ! file. open( String( "test-file.tnl" ), tnlWriteMode ) )
   {
      std::cerr << "Unable to create file test-file.tnl for the testing." << std::endl;
      return;
   }

   file. write< int, Devices::Cuda >( cudaIntData );
   file. write< float, Devices::Cuda, int >( cudaFloatData, 3 );
   if( ! file. close() )
   {
      std::cerr << "Unable to close the file test-file.tnl" << std::endl;
      return;
   }

   if( ! file. open( String( "test-file.tnl" ), tnlReadMode ) )
   {
      std::cerr << "Unable to open the file test-file.tnl for the testing." << std::endl;
      return;
   }
   int newIntData;
   float newFloatData[ 3 ];
   int* newCudaIntData;
   float* newCudaFloatData;
   cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
   file. read< int, Devices::Cuda >( newCudaIntData, 1 );
   file. read< float, Devices::Cuda, int >( newCudaFloatData, 3 );
   cudaMemcpy( &newIntData,
               newCudaIntData,
               sizeof( int ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( newFloatData,
               newCudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );

   ASSERT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      ASSERT_EQ( newFloatData[ i ], floatData[ i ] );
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
