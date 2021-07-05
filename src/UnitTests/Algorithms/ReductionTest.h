/***************************************************************************
                          ReductionTest.h  -  description
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
#include <TNL/Algorithms/Reduction.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST

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
   using Array = Containers::Array< int, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.setValue( 1 );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::Plus<>{} );
      EXPECT_EQ( res, size );
   }
}

TYPED_TEST( ReduceTest, min )
{
   using Array = Containers::Array< int, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::Min<>{} );
      EXPECT_EQ( res, 1 );
   }
}

TYPED_TEST( ReduceTest, max )
{
   using Array = Containers::Array< int, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, int& value ) { value = idx + 1;} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::Max<>{} );
      EXPECT_EQ( res, size );
   }
}


TYPED_TEST( ReduceTest, logicalAnd )
{
   using Array = Containers::Array< bool, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, bool& value ) { value = ( bool ) ( idx % 2 ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::LogicalAnd<>{} );
      EXPECT_EQ( res, false );
   }
}

TYPED_TEST( ReduceTest, logicalOr )
{
   using Array = Containers::Array< bool, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, bool& value ) { value = ( bool ) ( idx % 2 ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::LogicalOr<>{} );
      EXPECT_EQ( res, true );
   }
}

TYPED_TEST( ReduceTest, bitAnd )
{
   using Array = Containers::Array< char, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, char& value ) { value = 1 | ( 1 << ( idx % 8 ) ); } );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::BitAnd<>{} );
      EXPECT_EQ( res, 1 );
   }
}

TYPED_TEST( ReduceTest, bitOr )
{
   using Array = Containers::Array< char, Devices::Host >;
   Array a;
   for( int size = 100; size <= 1000000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( int idx, char& value ) { value = 1 << ( idx % 8 );} );
      auto a_view = a.getView();

      auto fetch = [=] __cuda_callable__ ( int idx ) { return a_view[ idx ]; };
      auto res = Algorithms::reduce< Devices::Host >( ( int ) 0, size, fetch, TNL::BitOr<>{} );
      EXPECT_EQ( res, ( char ) 255 );
   }
}

#endif

#include "../main.h"
