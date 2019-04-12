/***************************************************************************
                          ArrayTest.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// minimal custom data structure usable as ValueType in Array
struct MyData
{
   double data;

   __cuda_callable__
   MyData() : data(0) {}

   template< typename T >
   __cuda_callable__
   MyData( T v ) : data(v) {}

   __cuda_callable__
   bool operator==( const MyData& v ) const { return data == v.data; }

   // operator used in tests, not necessary for Array to work
   template< typename T >
   bool operator==( T v ) const { return data == v; }

   static String getType()
   {
      return String( "MyData" );
   }
};

std::ostream& operator<<( std::ostream& str, const MyData& v )
{
   return str << v.data;
}


// test fixture for typed tests
template< typename Array >
class ArrayTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ArrayTest is instantiated
using ArrayTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Array< int,    Devices::Host, short >,
   Array< long,   Devices::Host, short >,
   Array< float,  Devices::Host, short >,
   Array< double, Devices::Host, short >,
   Array< MyData, Devices::Host, short >,
   Array< int,    Devices::Host, int >,
   Array< long,   Devices::Host, int >,
   Array< float,  Devices::Host, int >,
   Array< double, Devices::Host, int >,
   Array< MyData, Devices::Host, int >,
   Array< int,    Devices::Host, long >,
   Array< long,   Devices::Host, long >,
   Array< float,  Devices::Host, long >,
   Array< double, Devices::Host, long >,
   Array< MyData, Devices::Host, long >
   // FIXME: this segfaults in String::~String()
//   Array< String, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   Array< int,    Devices::Cuda, short >,
   Array< long,   Devices::Cuda, short >,
   Array< float,  Devices::Cuda, short >,
   Array< double, Devices::Cuda, short >,
   Array< MyData, Devices::Cuda, short >,
   Array< int,    Devices::Cuda, int >,
   Array< long,   Devices::Cuda, int >,
   Array< float,  Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   Array< MyData, Devices::Cuda, int >,
   Array< int,    Devices::Cuda, long >,
   Array< long,   Devices::Cuda, long >,
   Array< float,  Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >,
   Array< MyData, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   Array< int,    Devices::MIC, short >,
   Array< long,   Devices::MIC, short >,
   Array< float,  Devices::MIC, short >,
   Array< double, Devices::MIC, short >,
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, short >,
   Array< int,    Devices::MIC, int >,
   Array< long,   Devices::MIC, int >,
   Array< float,  Devices::MIC, int >,
   Array< double, Devices::MIC, int >,
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, int >,
   Array< int,    Devices::MIC, long >,
   Array< long,   Devices::MIC, long >,
   Array< float,  Devices::MIC, long >,
   Array< double, Devices::MIC, long >
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, long >
#endif

   // all array tests should also work with Vector
   // (but we can't test all types because the argument list would be too long...)
#ifndef HAVE_CUDA
   ,
   Vector< float,  Devices::Host, long >,
   Vector< double, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   ,
   Vector< float,  Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   Vector< float,  Devices::MIC, long >,
   Vector< double, Devices::MIC, long >
#endif
>;

TYPED_TEST_SUITE( ArrayTest, ArrayTypes );


TYPED_TEST( ArrayTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   EXPECT_EQ( u.getSize(), 0 );

   ArrayType v( 10 );
   EXPECT_EQ( v.getSize(), 10 );

   // deep copy
   ArrayType w( v );
   EXPECT_NE( w.getData(), v.getData() );
   EXPECT_EQ( w.getSize(), v.getSize() );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), w.getElement( i ) );
   v.reset();
   EXPECT_EQ( w.getSize(), 10 );

   ArrayType a1 { 1, 2, 3 };
   EXPECT_EQ( a1.getElement( 0 ), 1 );
   EXPECT_EQ( a1.getElement( 1 ), 2 );
   EXPECT_EQ( a1.getElement( 2 ), 3 );

   std::list< int > l = { 4, 5, 6 };
   ArrayType a2( l );
   EXPECT_EQ( a2.getElement( 0 ), 4 );
   EXPECT_EQ( a2.getElement( 1 ), 5 );
   EXPECT_EQ( a2.getElement( 2 ), 6 );

   std::vector< int > q = { 7, 8, 9 };

   ArrayType a3( q );
   EXPECT_EQ( a3.getElement( 0 ), 7 );
   EXPECT_EQ( a3.getElement( 1 ), 8 );
   EXPECT_EQ( a3.getElement( 2 ), 9 );
}

TYPED_TEST( ArrayTest, setSize )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   const int maxSize = 10;
   for( int i = 0; i <= maxSize; i ++ ) {
      u.setSize( i );
      EXPECT_EQ( u.getSize(), i );
   }
}

TYPED_TEST( ArrayTest, empty )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType u( 10 );

   EXPECT_FALSE( u.empty() );
   u.reset();
   EXPECT_TRUE( u.empty() );
}

TYPED_TEST( ArrayTest, setLike )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 );
   EXPECT_EQ( u.getSize(), 10 );

   ArrayType v;
   v.setLike( u );
   EXPECT_EQ( v.getSize(), u.getSize() );
   EXPECT_NE( v.getData(), u.getData() );
}

TYPED_TEST( ArrayTest, swap )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 20 );
   u.setValue( 0 );
   v.setValue( 1 );
   u.swap( v );
   EXPECT_EQ( u.getSize(), 20 );
   EXPECT_EQ( v.getSize(), 10 );
   for( int i = 0; i < 20; i++ )
      EXPECT_EQ( u.getElement( i ), 1 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
}

TYPED_TEST( ArrayTest, reset )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   u.setSize( 100 );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_FALSE( u.empty() );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_TRUE( u.empty() );
   EXPECT_EQ( u.getData(), nullptr );
   u.setSize( 100 );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_FALSE( u.empty() );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_TRUE( u.empty() );
   EXPECT_EQ( u.getData(), nullptr );
}

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::Host, Index >&& u )
{
   u.setSize( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
   }
}

#ifdef HAVE_CUDA
template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ValueType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getSize() )
      ( *u )[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::Cuda, Index >&& u )
{
#ifdef HAVE_CUDA
   u.setSize( 10 );
   using ArrayType = Array< Value, Devices::Cuda, Index >;
   ArrayType* kernel_u = Devices::Cuda::passToDevice( u );
   testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
   Devices::Cuda::freeFromDevice( kernel_u );
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( u.getElement( i ), i );
   }
#endif
}

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::MIC, Index >&& u )
{
#ifdef HAVE_MIC
   // TODO
#endif
}

TYPED_TEST( ArrayTest, elementwiseAccess )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayElementwiseAccess( ArrayType() );
}

TYPED_TEST( ArrayTest, containsValue )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType array;
   array.setSize( 1024 );

   for( int i = 0; i < array.getSize(); i++ )
      array.setElement( i, i % 10 );

   for( int i = 0; i < 10; i++ )
      EXPECT_TRUE( array.containsValue( i ) );

   for( int i = 10; i < 20; i++ )
      EXPECT_FALSE( array.containsValue( i ) );
}

TYPED_TEST( ArrayTest, containsOnlyValue )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType array;
   array.setSize( 1024 );

   for( int i = 0; i < array.getSize(); i++ )
      array.setElement( i, i % 10 );

   for( int i = 0; i < 20; i++ )
      EXPECT_FALSE( array.containsOnlyValue( i ) );

   array.setValue( 100 );
   EXPECT_TRUE( array.containsOnlyValue( 100 ) );
}

TYPED_TEST( ArrayTest, comparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 10 ), w( 10 );
   typename ArrayType::HostType u_host( 10 );
   for( int i = 0; i < 10; i ++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
      v.setElement( i, i );
      w.setElement( i, 2 * i );
   }
   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   // comparison with different device
   EXPECT_TRUE( u == u_host );
   EXPECT_TRUE( u_host == u );
   EXPECT_TRUE( w != u_host );
   EXPECT_TRUE( u_host != w );

   v.setSize( 0 );
   EXPECT_FALSE( u == v );
   u.setSize( 0 );
   EXPECT_TRUE( u == v );
}

TYPED_TEST( ArrayTest, comparisonOperatorWithDifferentType )
{
   using DeviceType = typename TestFixture::ArrayType::DeviceType;
   using ArrayType1 = Array< short, DeviceType >;
   using ArrayType2 = Array< float, Devices::Host >;

   ArrayType1 u( 10 );
   ArrayType2 v( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      v.setElement( i, i );
   }
   EXPECT_TRUE( u == v );
   EXPECT_FALSE( u != v );

   // the comparison will be in floats
   v.setElement( 0, 0.1f );
   EXPECT_FALSE( u == v );
   EXPECT_TRUE( u != v );
}

TYPED_TEST( ArrayTest, assignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 10 );
   typename ArrayType::HostType u_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
   }

   v = 42;
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 42 );
   v = u;
   EXPECT_EQ( u, v );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( u, v );

   // assignment from device to host
   u_host.setValue( 0 );
   u_host = u;
   EXPECT_EQ( u_host, u );

   u = 5;
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( u.getElement( i ), 5 );
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ValueType >::value >::type >
void testArrayAssignmentWithDifferentType()
{
   ArrayType u( 10 );
   Array< short, typename ArrayType::DeviceType, short > v( 10 );
   Array< short, Devices::Host, short > v_host( 10 );
   typename ArrayType::HostType u_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
   }

   v.setValue( 0 );
   v = u;
   EXPECT_EQ( v, u );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( v, u_host );

   // assignment from device to host
   v_host.setValue( 0 );
   v_host = u;
   EXPECT_EQ( v_host, u );
}

template< typename ArrayType,
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ValueType >::value >::type,
          typename = void >
void testArrayAssignmentWithDifferentType()
{
}

TYPED_TEST( ArrayTest, assignmentOperatorWithDifferentType )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayAssignmentWithDifferentType< ArrayType >();
}

TYPED_TEST( ArrayTest, SaveAndLoad )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u, v;
   v.setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v.setElement( i, 3.14147 );
   File file;
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::Out ) );
   ASSERT_NO_THROW( v.save( file ) );
   ASSERT_NO_THROW( file.close() );
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::In ) );
   ASSERT_NO_THROW( u.load( file ) );
   EXPECT_EQ( u, v );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

TYPED_TEST( ArrayTest, boundLoad )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType v, w;
   v.setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v.setElement( i, 3.14147 );
   File file;
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::Out ) );
   ASSERT_NO_THROW( v.save( file ) );
   ASSERT_NO_THROW( file.close() );

   w.setSize( 100 );
   auto u = w.getView();
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::In ) );
   ASSERT_NO_THROW( u.load( file ) );
   EXPECT_EQ( u, v );
   EXPECT_EQ( u.getData(), w.getData() );

   ArrayType z( 50 );
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::In ) );
   EXPECT_ANY_THROW( z.boundLoad( file ) );

   v.reset();
   ASSERT_NO_THROW( file.open( "test-file.tnl", File::Mode::In ) );
   EXPECT_NO_THROW( v.boundLoad( file ) );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

// TODO: test all __cuda_callable__ methods from a CUDA kernel

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
