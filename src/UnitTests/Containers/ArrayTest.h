/***************************************************************************
                          ArrayTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ElementType, Devices::Cuda, IndexType >* u );
#endif

class testingClassForArrayTester
{
   public:

      static String getType()
      {
         return String( "testingClassForArrayTester" );
      };
};

String getType( const testingClassForArrayTester& c )
{
   return String( "testingClassForArrayTester" );
};

#ifdef HAVE_GTEST


// TODO: Fix this
#if GTEST_HAS_TYPED_TEST

using ::testing::Types;
typedef Types< int, Devices::Host, int > MyTypes;
/*TYPED_TEST_CASE( ArrayTest, MyTypes );

TYPED_TEST( ArrayTest, testConstructorDestructor )
{
   typedef Array< TypeParam > ArrayType;
   ArrayType u;
   ArrayType v( 10 );
   ASSERT_EQ( v.getSize(), 10 );
}

TYPED_TEST( ArrayTest, testSetSize )
{
   typedef Array< TypeParam > ArrayType;
   ArrayType u, v;
   u.setSize( 10 );
   v.setSize( 10 );
   ASSERT_EQ( u.getSize(), 10 );
   ASSERT_EQ( v.getSize(), 10 );
}

TYPED_TEST( ArrayTest, testBind )
{
   typedef Array< TypeParam > ArrayType;
   ArrayType u( 10 ), v;
   u.setValue( 27 );
   v.bind( u );
   ASSERT_EQ( v.getSize(), u.getSize() );
   ASSERT_EQ( u.getElement( 0 ), 27 );
   v.setValue( 50 );
   ASSERT_EQ( u.getElement( 0 ), 50 );
   u.reset();
   ASSERT_EQ( u.getSize(), 0 );
   ASSERT_EQ( v.getElement( 0 ), 50 );

   ElementType data[ 10 ] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
   u.bind( data, 10 );
   ASSERT_EQ( u.getElement( 1 ), 2 );
   v.bind( u );
   ASSERT_EQ( v.getElement( 1 ), 2 );
   u.reset();
   v.setElement( 1, 3 );
   v.reset();
   ASSERT_EQ( data[ 1 ], 3 );
}*/

#endif  /* GTEST_HAS_TYPED_TEST */

typedef int ElementType;
typedef Devices::Host Device;
typedef int IndexType;

TEST( ArrayTest, testSetGetElement )
{
   using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > u;
   u. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      u. setElement( i, i );
   for( int i = 0; i < 10; i ++ )
      ASSERT_EQ( u. getElement( i ), i );

   u.setValue( 0 );
   if( std::is_same< Device, Devices::Host >::value )
   {
      for( int i = 0; i < 10; i ++ )
         u[ i ] =  i;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Array< ElementType, Device, IndexType >* kernel_u =
               Devices::Cuda::passToDevice( u );
      testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
      Devices::Cuda::freeFromDevice( kernel_u );
      ASSERT_TRUE( TNL_CHECK_CUDA_DEVICE );
#endif
   }
   for( int i = 0; i < 10; i++ )
      ASSERT_EQ( u.getElement( i ), i );
};

TEST( ArrayTest, testComparisonOperator )
{
    using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > u;
   Array< ElementType, Device, IndexType > v;
   Array< ElementType, Device, IndexType > w;
   u. setSize( 10 );
   v. setSize( 10 );
   w. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      u. setElement( i, i );
      v. setElement( i, i );
      w. setElement( i, 2*1 );
   }
   ASSERT_TRUE( u == v );
   ASSERT_FALSE( u != v );
   ASSERT_TRUE( u != w );
   ASSERT_FALSE( u == w );
};

TEST( ArrayTest, testAssignmentOperator )
{
   using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > u;
   Array< ElementType, Device, IndexType > v;
   u. setSize( 10 );
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      u. setElement( i, i );
   v = u;
   ASSERT_TRUE( u == v );
   ASSERT_TRUE( v == u );
   ASSERT_FALSE( u != v );
   ASSERT_FALSE( v != u );

   v.setValue( 0 );
   Array< ElementType, Devices::Host, IndexType > w;
   w.setSize( 10 );
   w = u;

   ASSERT_TRUE( u == w );
   ASSERT_FALSE( u != w );

   v.setValue( 0 );
   v = w;
   ASSERT_TRUE( v == w );
   ASSERT_FALSE( v != w );
};

TEST( ArrayTest, testGetSize )
{
   using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > u;
   const int maxSize = 10;
   for( int i = 0; i < maxSize; i ++ )
      u. setSize( i );

   ASSERT_EQ( u. getSize(), maxSize - 1 );
};

TEST( ArrayTest, testReset )
{
   using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > u;
   u. setSize( 100 );
   ASSERT_EQ( u. getSize(), 100 );
   u. reset();
   ASSERT_EQ( u. getSize(), 0 );
   u. setSize( 100 );
   ASSERT_EQ( u. getSize(), 100 );
   u. reset();
   ASSERT_EQ( u. getSize(), 0 );

};

TEST( ArrayTest, testSetSizeAndDestructor )
{
   using namespace TNL::Containers;
   for( int i = 0; i < 100; i ++ )
   {
      Array< ElementType, Device, IndexType > u;
      u. setSize( i );
   }
}

TEST( ArrayTest, testSaveAndLoad )
{
   using namespace TNL::Containers;
   Array< ElementType, Device, IndexType > v;
   v. setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v. setElement( i, 3.14147 );
   File file;
   file. open( "test-file.tnl", IOMode::write );
   v. save( file );
   file. close();
   Array< ElementType, Device, IndexType > u;
   file. open( "test-file.tnl", IOMode::read );
   u. load( file );
   file. close();
   ASSERT_TRUE( u == v );
}

TEST( ArrayTest, testUnusualStructures )
{
   using namespace TNL::Containers;
   Array< testingClassForArrayTester >u;
};

#endif /* HAVE_GTEST */
   
#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ElementType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getSize() )
      ( *u )[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */


int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}


