/***************************************************************************
                          reduceTest.h  -  description
                             -------------------
    begin                : Jul 2, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/reduce.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST

template< typename Device >
void ReduceTest_sum()
{
   using Array = Containers::Array< int, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.setValue( 1 );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::Plus{} );
      EXPECT_EQ( res, size );
   }
}

template< typename Device >
void ReduceTest_min()
{
   using Array = Containers::Array< int, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::Min{} );
      EXPECT_EQ( res, 1 );
   }
}

template< typename Device >
void ReduceTest_max()
{
   using Array = Containers::Array< int, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::Max{} );
      EXPECT_EQ( res, size );
   }
}

template< typename Device >
void ReduceTest_minWithArg()
{
   using Array = Containers::Array< int, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduceWithArgument< Device >( ( int ) 0, size, fetch, TNL::MinWithArg{} );
      EXPECT_EQ( res.first, 1 );
      EXPECT_EQ( res.second, 0 );
   }
}

template< typename Device >
void ReduceTest_maxWithArg()
{
   using Array = Containers::Array< int, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduceWithArgument< Device >( ( int ) 0, size, fetch, TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, size );
      EXPECT_EQ( res.second, size - 1 );
   }
}

template< typename Device >
void ReduceTest_logicalAnd()
{
   using Array = Containers::Array< bool, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, bool& value ) { value = ( bool ) ( idx % 2 ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::LogicalAnd{} );
      EXPECT_EQ( res, false );
   }
}

template< typename Device >
void ReduceTest_logicalOr()
{
   using Array = Containers::Array< bool, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, bool& value ) { value = ( bool ) ( idx % 2 ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::LogicalOr{} );
      EXPECT_EQ( res, true );
   }
}

template< typename Device >
void ReduceTest_bitAnd()
{
   using Array = Containers::Array< char, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, char& value ) { value = 1 | ( 1 << ( idx % 8 ) ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::BitAnd{} );
      EXPECT_EQ( res, 1 );
   }
}

template< typename Device >
void ReduceTest_bitOr()
{
   using Array = Containers::Array< char, Device >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, char& value ) { value = 1 << ( idx % 8 );} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Device >( ( int ) 0, size, fetch, TNL::BitOr{} );
      EXPECT_EQ( res, ( char ) 255 );
   }
}

// test fixture for typed tests
template< typename Device >
class ReduceTest : public ::testing::Test
{
protected:
   using DeviceType = Device;
};

// types for which ArrayTest is instantiated
using DeviceTypes = ::testing::Types<
   Devices::Host
#ifdef HAVE_CUDA
   ,Devices::Cuda
#endif
   >;

TYPED_TEST_SUITE( ReduceTest, DeviceTypes );

TYPED_TEST( ReduceTest, sum )
{
   ReduceTest_sum< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, min )
{
   ReduceTest_min< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, max )
{
   ReduceTest_max< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, minWithArg )
{
   ReduceTest_minWithArg< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, maxWithArg )
{
   ReduceTest_maxWithArg< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, logicalAnd )
{
   ReduceTest_logicalAnd< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, logicalOr )
{
   ReduceTest_logicalOr< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, bitAnd )
{
   ReduceTest_bitAnd< typename TestFixture::DeviceType >();
}

TYPED_TEST( ReduceTest, bitOr )
{
   ReduceTest_bitOr< typename TestFixture::DeviceType >();
}

#endif

#include "../main.h"
