/***************************************************************************
                          ArrayOperationsTester.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/ArrayOperations.h>
#include <TNL/Devices/Cuda.h>

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

using namespace TNL;
using namespace TNL::Containers;

int getTestSize()
{
   return 1 << 16;
   //const int cudaGridSize = 256;
   //return 1.5 * cudaGridSize * maxCudaBlockSize;
   //return  1 << 22;
};

typedef int Element;

#ifdef HAVE_GTEST

TEST( ArrayOperationsTest, allocationTest )
{
   Element* data;
   ArrayOperations< Devices::Host >::allocateMemory( data, getTestSize() );
   ASSERT_EQ( data, ( Element* ) NULL );

   ArrayOperations< Devices::Host >::freeMemory( data );
};

TEST( ArrayOperationsTest, memorySetTest )
{
   const int size = 1024;
   Element *data;
   ArrayOperations< Devices::Host > :: allocateMemory( data, size );
   ArrayOperations< Devices::Host > :: setMemory( data, 13, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( data[ i ], 13 );
   ArrayOperations< Devices::Host > :: freeMemory( data );
};

TEST( ArrayOperationsTest, copyMemoryTest )
{
   const int size = getTestSize();

   Element *data1, *data2;
   ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
   ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
   ArrayOperations< Devices::Host > :: setMemory( data1, 13, size );
   ArrayOperations< Devices::Host > :: copyMemory< Element, Element, int >( data2, data1, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( data1[ i ], data2[ i ]);
   ArrayOperations< Devices::Host > :: freeMemory( data1 );
   ArrayOperations< Devices::Host > :: freeMemory( data2 );
};

TEST( ArrayOperationsTest, copyMemoryWithConversionTest )
{
   const int size = getTestSize();
   int *data1;
   float *data2;
   ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
   ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
   ArrayOperations< Devices::Host > :: setMemory( data1, 13, size );
   ArrayOperations< Devices::Host > :: copyMemory< float, int, int >( data2, data1, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( data1[ i ], data2[ i ] );
   ArrayOperations< Devices::Host > :: freeMemory( data1 );
   ArrayOperations< Devices::Host > :: freeMemory( data2 );
};

TEST( ArrayOperationsTest, compareMemoryTest )
{
   const int size = getTestSize();
   int *data1, *data2;
   ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
   ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
   ArrayOperations< Devices::Host > :: setMemory( data1, 7, size );
   ASSERT_FALSE( ( ArrayOperations< Devices::Host > :: compareMemory< int, int, int >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host > :: setMemory( data2, 7, size );
   ASSERT_TRUE( ( ArrayOperations< Devices::Host > :: compareMemory< int, int, int >( data1, data2, size ) ) );
};

TEST( ArrayOperationsTest, compareMemoryWithConversionTest )
{
   const int size = getTestSize();
   int *data1;
   float *data2;
   ArrayOperations< Devices::Host > :: allocateMemory( data1, size );
   ArrayOperations< Devices::Host > :: allocateMemory( data2, size );
   ArrayOperations< Devices::Host > :: setMemory( data1, 7, size );
   ASSERT_FALSE( ( ArrayOperations< Devices::Host > :: compareMemory< int, float, int >( data1, data2, size ) ) );
   ArrayOperations< Devices::Host > :: setMemory( data2, ( float ) 7.0, size );
   ASSERT_TRUE( ( ArrayOperations< Devices::Host > :: compareMemory< int, float, int >( data1, data2, size ) ) );
};


#ifdef HAVE_CUDA
TEST( ArrayOperationsTest, allocationTest )
{
   int* data;
   ArrayOperations< Devices::Cuda >::allocateMemory( data, getTestSize() );
   ASSERT_TRUE( checkCudaDevice );

   ArrayOperations< Devices::Cuda >::freeMemory( data );
   ASSERT_TRUE( checkCudaDevice );
}

TEST( ArrayOperationsTest, setMemoryElementTest )
{
   const int size( 1024 );
   int* data;
   ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
   ASSERT_TRUE( checkCudaDevice );

   for( int i = 0; i < getTestSize(); i++ )
      ArrayOperations< Devices::Cuda >::setMemoryElement( &data[ i ], i );

   for( int i = 0; i < size; i++ )
   {
      int d;
      ASSERT_EQ( cudaMemcpy( &d, &data[ i ], sizeof( int ), cudaMemcpyDeviceToHost ), cudaSuccess );
      ASSERT_EQ( d, i );
   }

   ArrayOperations< Devices::Cuda >::freeMemory( data );
   ASSERT_TRUE( checkCudaDevice );
}

TEST( ArrayOperationsTest, getMemoryElementTest )
{
   const int size( 1024 );
   int* data;
   ArrayOperations< Devices::Cuda >::allocateMemory( data, size );
   ASSERT_TRUE( checkCudaDevice );

   for( int i = 0; i < getTestSize(); i++ )
      ArrayOperations< Devices::Cuda >::setMemoryElement( &data[ i ], i );

   for( int i = 0; i < size; i++ )
      ASSERT_EQ( ( ArrayOperations< Devices::Cuda >::getMemoryElement( &data[ i ] ), i ) );

   ArrayOperations< Devices::Cuda >::freeMemory( data );
   ASSERT_TRUE( checkCudaDevice );
}


TEST( ArrayOperationsTest, smallMemorySetTest )
{
   const int size = 1024;
   int *hostData, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, 0, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, 13, size );
   ASSERT_TRUE( checkCudaDevice );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData, deviceData, size );
   ASSERT_TRUE( checkCudaDevice );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( hostData[ i ], 13 );
   ArrayOperations< Devices::Cuda >::freeMemory( hostData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
};

TEST( ArrayOperationsTest, bigMemorySetTest )
{
   const int size( getTestSize() );
   int *hostData, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, 0, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, 13, size );
   ASSERT_TRUE( checkCudaDevice );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData, deviceData, size );
   ASSERT_TRUE( checkCudaDevice );
   for( int i = 0; i < size; i += 100 )
   {
      if( hostData[ i ] != 13 )
      ASSERT_EQ( hostData[ i ], 13 );
   }
   ArrayOperations< Devices::Host >::freeMemory( hostData );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
};

TEST( ArrayOperationsTest, copyMemoryTest )
{
   const int size = getTestSize();

   int *hostData1, *hostData2, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int >( deviceData, hostData1, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< int, int >( hostData2, deviceData, size );
   ASSERT_TRUE( ( ArrayOperations< Devices::Host >::compareMemory< int, int >( hostData1, hostData2, size) ) );
   ArrayOperations< Devices::Host >::freeMemory( hostData1 );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
};

TEST( ArrayOperationsTest, copyMemoryWithConversionHostToCudaTest )
{
   const int size = getTestSize();
   int *hostData1;
   float *hostData2, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< float, int, int >( deviceData, hostData1, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, float, int >( hostData2, deviceData, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( hostData1[ i ], hostData2[ i ] );
   ArrayOperations< Devices::Host >::freeMemory( hostData1 );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
};

TEST( ArrayOperationsTest, copyMemoryWithConversionCudaToHostTest )
{
   const int size = getTestSize();
   int *hostData1, *deviceData;
   float *hostData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int >( deviceData, hostData1, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, int, int >( hostData2, deviceData, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( hostData1[ i ], hostData2[ i ] );
   ArrayOperations< Devices::Host >::freeMemory( hostData1 );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData );
};

TEST( ArrayOperationsTest, copyMemoryWithConversionCudaToCudaTest )
{
   const int size = getTestSize();
   int *hostData1, *deviceData1;
   float *hostData2, *deviceData2;
   ArrayOperations< Devices::Host >::allocateMemory( hostData1, size );
   ArrayOperations< Devices::Host >::allocateMemory( hostData2, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData1, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData2, size );
   ArrayOperations< Devices::Host >::setMemory( hostData1, 13, size );
   ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< int, int, int >( deviceData1, hostData1, size );
   ArrayOperations< Devices::Cuda >::copyMemory< float, int, int >( deviceData2, deviceData1, size );
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< float, float, int >( hostData2, deviceData2, size );
   for( int i = 0; i < size; i ++ )
      ASSERT_EQ( hostData1[ i ], hostData2[ i ] );
   ArrayOperations< Devices::Host >::freeMemory( hostData1 );
   ArrayOperations< Devices::Host >::freeMemory( hostData2 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData1 );
   ArrayOperations< Devices::Cuda >::freeMemory( deviceData2 );
};

TEST( ArrayOperationsTest, compareMemoryHostCudaTest )
{
   const int size = getTestSize();
   int *hostData, *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, 8, size );
   ASSERT_FALSE( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, 7, size );
   ASSERT_TRUE( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, int, int >( hostData, deviceData, size ) ) );
};

TEST( ArrayOperationsTest, compareMemoryWithConversionHostCudaTest )
{
   const int size = getTestSize();
   int *hostData;
   float *deviceData;
   ArrayOperations< Devices::Host >::allocateMemory( hostData, size );
   ArrayOperations< Devices::Cuda >::allocateMemory( deviceData, size );
   ArrayOperations< Devices::Host >::setMemory( hostData, 7, size );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, ( float ) 8.0, size );
   ASSERT_FALSE( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
   ArrayOperations< Devices::Cuda >::setMemory( deviceData, ( float ) 7.0, size );
   ASSERT_TRUE( ( ArrayOperations< Devices::Host, Devices::Cuda >::compareMemory< int, float, int >( hostData, deviceData, size ) ) );
};
#endif // HAVE_CUDA
#endif // HAVE_GTEST

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}



