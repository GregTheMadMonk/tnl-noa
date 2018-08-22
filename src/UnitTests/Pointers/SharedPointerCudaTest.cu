/***************************************************************************
                          SharedPointerCudaTest.cpp  -  description
                             -------------------
    begin                : Aug 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Array.h>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/Devices/Cuda.h>
#include "../GtestMissingError.h"

using namespace TNL;

#ifdef HAVE_GTEST
TEST( SharedPointerCudaTest, ConstructorTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   Pointers::SharedPointer< TestType, Devices::Cuda > ptr1;

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   Pointers::SharedPointer< TestType, Devices::Cuda > ptr2( 1, 2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
#endif
};

TEST( SharedPointerCudaTest, getDataTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   Pointers::SharedPointer< TestType, Devices::Cuda > ptr1( 1, 2 );
   
#ifdef HAVE_CUDA_UNIFIED_MEMORY
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
#else
   
   Devices::Cuda::synchronizeDevice();
   
   TestType aux;
   
   cudaMemcpy( ( void*) &aux, &ptr1.getData< Devices::Cuda >(), sizeof( TestType ), cudaMemcpyDeviceToHost );
   
   ASSERT_EQ( aux[ 0 ], 1 );
   ASSERT_EQ( aux[ 1 ], 2 );
#endif  // HAVE_CUDA_UNIFIED_MEMORY
#endif  // HAVE_CUDA
};

#ifdef HAVE_CUDA
__global__ void copyArrayKernel( const TNL::Containers::Array< int, Devices::Cuda >* inArray,
                                 int* outArray )
{
   if( threadIdx.x < 2 )
   {
      outArray[ threadIdx.x ] = ( *inArray )[ threadIdx.x ];
   }
}

#endif

TEST( SharedPointerCudaTest, getDataArrayTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::Array< int, Devices::Cuda  > TestType;
   Pointers::SharedPointer< TestType > ptr;
   
   ptr->setSize( 2 );
   ptr->setElement( 0, 1 );
   ptr->setElement( 1, 2 );

   Devices::Cuda::synchronizeDevice();

   int *testArray_device, *testArray_host;
   cudaMalloc( ( void** ) &testArray_device, 2 * sizeof( int ) );
   copyArrayKernel<<< 1, 2 >>>( &ptr.getData< Devices::Cuda >(), testArray_device );
   testArray_host = new int [ 2 ];
   cudaMemcpy( testArray_host, testArray_device, 2 * sizeof( int ), cudaMemcpyDeviceToHost );
   
   ASSERT_EQ( testArray_host[ 0 ], 1 );
   ASSERT_EQ( testArray_host[ 1 ], 2 );
   
   delete[] testArray_host;
   cudaFree( testArray_device );

#endif
};


#endif

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
