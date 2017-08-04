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

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// minimal custom data structure usable as ElementType in Array
struct MyData
{
   double data;

   __cuda_callable__
   MyData() : data(0) {}

   template< typename T >
   __cuda_callable__
   MyData( T v ) : data(v) {}

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
   Array< short,  Devices::Host, short >,
   Array< int,    Devices::Host, short >,
   Array< long,   Devices::Host, short >,
   Array< float,  Devices::Host, short >,
   Array< double, Devices::Host, short >,
   Array< MyData, Devices::Host, short >,
   Array< short,  Devices::Host, int >,
   Array< int,    Devices::Host, int >,
   Array< long,   Devices::Host, int >,
   Array< float,  Devices::Host, int >,
   Array< double, Devices::Host, int >,
   Array< MyData, Devices::Host, int >,
   Array< short,  Devices::Host, long >,
   Array< int,    Devices::Host, long >,
   Array< long,   Devices::Host, long >,
   Array< float,  Devices::Host, long >,
   Array< double, Devices::Host, long >,
   Array< MyData, Devices::Host, long >
   // FIXME: this segfaults in String::~String()
//   , Array< String, Devices::Host, long >
#ifdef HAVE_CUDA
   ,
   Array< short,  Devices::Cuda, short >,
   Array< int,    Devices::Cuda, short >,
   Array< long,   Devices::Cuda, short >,
   Array< float,  Devices::Cuda, short >,
   Array< double, Devices::Cuda, short >,
   Array< MyData, Devices::Cuda, short >,
   Array< short,  Devices::Cuda, int >,
   Array< int,    Devices::Cuda, int >,
   Array< long,   Devices::Cuda, int >,
   Array< float,  Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   Array< MyData, Devices::Cuda, int >,
   Array< short,  Devices::Cuda, long >,
   Array< int,    Devices::Cuda, long >,
   Array< long,   Devices::Cuda, long >,
   Array< float,  Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >,
   Array< MyData, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   Array< short,  Devices::MIC, short >,
   Array< int,    Devices::MIC, short >,
   Array< long,   Devices::MIC, short >,
   Array< float,  Devices::MIC, short >,
   Array< double, Devices::MIC, short >,
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, short >,
   Array< short,  Devices::MIC, int >,
   Array< int,    Devices::MIC, int >,
   Array< long,   Devices::MIC, int >,
   Array< float,  Devices::MIC, int >,
   Array< double, Devices::MIC, int >,
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, int >,
   Array< short,  Devices::MIC, long >,
   Array< int,    Devices::MIC, long >,
   Array< long,   Devices::MIC, long >,
   Array< float,  Devices::MIC, long >,
   Array< double, Devices::MIC, long >
   // TODO: MyData does not work on MIC
//   Array< MyData, Devices::MIC, long >
#endif
>;

TYPED_TEST_CASE( ArrayTest, ArrayTypes );


TYPED_TEST( ArrayTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   EXPECT_EQ( u.getSize(), 0 );

   ArrayType v( 10 );
   EXPECT_EQ( v.getSize(), 10 );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ];
      ArrayType w( data, 10 );
      EXPECT_EQ( w.getData(), data );

      ArrayType z1( w );
      EXPECT_EQ( z1.getData(), data );
      EXPECT_EQ( z1.getSize(), 10 );

      ArrayType z2( w, 1 );
      EXPECT_EQ( z2.getData(), data + 1 );
      EXPECT_EQ( z2.getSize(), 9 );

      ArrayType z3( w, 2, 3 );
      EXPECT_EQ( z3.getData(), data + 2 );
      EXPECT_EQ( z3.getSize(), 3 );
   }
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

   ArrayType v( u );
   EXPECT_EQ( v.getSize(), 10 );
   EXPECT_EQ( v.getData(), u.getData() );
   v.setSize( 11 );
   EXPECT_EQ( u.getSize(), 10 );
   EXPECT_EQ( v.getSize(), 11 );
   EXPECT_NE( v.getData(), u.getData() );

   // cast to bool returns true iff size > 0
   EXPECT_TRUE( (bool) u );
   EXPECT_FALSE( ! u );
   u.setSize( 0 );
   EXPECT_FALSE( (bool) u );
   EXPECT_TRUE( ! u );
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

TYPED_TEST( ArrayTest, bind )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v;
   v.bind( u );
   EXPECT_EQ( v.getSize(), u.getSize() );
   EXPECT_EQ( v.getData(), u.getData() );

   // bind array with offset and size
   ArrayType w;
   w.bind( u, 2, 3 );
   EXPECT_EQ( w.getSize(), 3 );
   EXPECT_EQ( w.getData(), u.getData() + 2 );

   // setting values
   u.setValue( 27 );
   EXPECT_EQ( u.getElement( 0 ), 27 );
   v.setValue( 50 );
   EXPECT_EQ( u.getElement( 0 ), 50 );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( v.getElement( 0 ), 50 );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
      u.bind( data, 10 );
      EXPECT_EQ( u.getData(), data );
      EXPECT_EQ( u.getSize(), 10 );
      EXPECT_EQ( u.getElement( 1 ), 2 );
      v.bind( u );
      EXPECT_EQ( v.getElement( 1 ), 2 );
      u.reset();
      v.setElement( 1, 3 );
      v.reset();
      EXPECT_EQ( data[ 1 ], 3 );
   }
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
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
   u.setSize( 100 );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
}

template< typename Element, typename Index >
void testArrayElementwiseAccess( Array< Element, Devices::Host, Index >&& u )
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
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ElementType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getSize() )
      ( *u )[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

template< typename Element, typename Index >
void testArrayElementwiseAccess( Array< Element, Devices::Cuda, Index >&& u )
{
#ifdef HAVE_CUDA
   u.setSize( 10 );
   using ArrayType = Array< Element, Devices::Cuda, Index >;
   ArrayType* kernel_u = Devices::Cuda::passToDevice( u );
   testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
   Devices::Cuda::freeFromDevice( kernel_u );
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( u.getElement( i ), i );
   }
#endif
}

template< typename Element, typename Index >
void testArrayElementwiseAccess( Array< Element, Devices::MIC, Index >&& u )
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

// TODO: comparison with different device
TYPED_TEST( ArrayTest, comparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 10 ), w( 10 );
   for( int i = 0; i < 10; i ++ ) {
      u.setElement( i, i );
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

   v.setSize( 0 );
   EXPECT_FALSE( u == v );
   u.setSize( 0 );
   EXPECT_TRUE( u == v );
}

// TODO: comparison with different device
// TODO: missing implementation of relevant reduction operation on CUDA with different types
/*
TYPED_TEST( ArrayTest, comparisonOperatorWithDifferentType )
{
   Array< short, typename ArrayType::DeviceType, short > z( 10 );
   for( int i = 0; i < 10; i ++ )
      z.setElement( i, i );
   EXPECT_TRUE( u == z );
   EXPECT_FALSE( u != z );
   for( int i = 0; i < 10; i ++ )
      z.setElement( i, 2 * i );
   EXPECT_FALSE( u == z );
   EXPECT_TRUE( u != z );
}
*/

// TODO: assignment from different device
TYPED_TEST( ArrayTest, assignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 10 );
   for( int i = 0; i < 10; i++ )
      u.setElement( i, i );
   v.setValue( 0 );
   v = u;
   EXPECT_EQ( u, v );
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ElementType >::value >::type >
void testArrayAssignmentWithDifferentType()
{
   ArrayType u( 10 );
   for( int i = 0; i < 10; i++ )
      u.setElement( i, i );
   Array< short, typename ArrayType::DeviceType, short > v( 10 );
   v.setValue( 0 );
   v = u;
// TODO: missing implementation of relevant reduction operation on CUDA with different types
//   EXPECT_EQ( u, v );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), i );
}

template< typename ArrayType,
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ElementType >::value >::type,
          typename = void >
void testArrayAssignmentWithDifferentType()
{
}

// TODO: assignment from different device
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
   file.open( "test-file.tnl", IOMode::write );
   EXPECT_TRUE( v.save( file ) );
   file.close();
   file.open( "test-file.tnl", IOMode::read );
   EXPECT_TRUE( u.load( file ) );
   EXPECT_EQ( u, v );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

TYPED_TEST( ArrayTest, boundLoad )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u, v, w;
   v.setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v.setElement( i, 3.14147 );
   File file;
   file.open( "test-file.tnl", IOMode::write );
   EXPECT_TRUE( v.save( file ) );
   file.close();

   w.setSize( 100 );
   u.bind( w );
   file.open( "test-file.tnl", IOMode::read );
   EXPECT_TRUE( u.boundLoad( file ) );
   EXPECT_EQ( u, v );
   EXPECT_EQ( u.getData(), w.getData() );

   u.setSize( 50 );
   file.open( "test-file.tnl", IOMode::read );
   EXPECT_FALSE( u.boundLoad( file ) );

   u.reset();
   file.open( "test-file.tnl", IOMode::read );
   EXPECT_TRUE( u.boundLoad( file ) );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

TYPED_TEST( ArrayTest, referenceCountingConstructors )
{
   using ArrayType = typename TestFixture::ArrayType;

   // copies of a dynamic array
   ArrayType u( 10 );
   ArrayType v( u );
   ArrayType w( v );
   EXPECT_EQ( v.getData(), u.getData() );
   EXPECT_EQ( w.getData(), u.getData() );

   // copies of a static array
   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      ArrayType u( data, 10 );
      ArrayType v( u );
      ArrayType w( v );
      EXPECT_EQ( u.getData(), data );
      EXPECT_EQ( v.getData(), data );
      EXPECT_EQ( w.getData(), data );
   }
}

TYPED_TEST( ArrayTest, referenceCountingBind )
{
   using ArrayType = typename TestFixture::ArrayType;

   // copies of a dynamic array
   ArrayType u( 10 );
   ArrayType v;
   v.bind( u );
   ArrayType w;
   w.bind( v );
   EXPECT_EQ( v.getData(), u.getData() );
   EXPECT_EQ( w.getData(), u.getData() );

   // copies of a static array
   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      ArrayType u( data, 10 );
      ArrayType v;
      v.bind( u );
      ArrayType w;
      w.bind( v );
      EXPECT_EQ( u.getData(), data );
      EXPECT_EQ( v.getData(), data );
      EXPECT_EQ( w.getData(), data );
   }
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
