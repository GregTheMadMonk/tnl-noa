/***************************************************************************
                          ArrayViewTest.h -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST 
#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

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
class ArrayViewTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ViewType = ArrayView< typename Array::ElementType, typename Array::DeviceType, typename Array::IndexType >;
};

// types for which ArrayViewTest is instantiated
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

TYPED_TEST_CASE( ArrayViewTest, ArrayTypes );


TYPED_TEST( ArrayViewTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ConstViewType = ArrayView< const typename ArrayType::ElementType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;

   ArrayType a( 10 );
   EXPECT_EQ( a.getSize(), 10 );

   ViewType v( a );
   EXPECT_EQ( v.getSize(), 10 );
   EXPECT_EQ( v.getData(), a.getData() );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ];
      ViewType w( data, 10 );
      EXPECT_EQ( w.getData(), data );

      ViewType z( w );
      EXPECT_EQ( z.getData(), data );
      EXPECT_EQ( z.getSize(), 10 );
   }

   // test initialization by const reference
   const ArrayType& b = a;
   ConstViewType b_view( b );
   ConstViewType const_a_view( a );
}

TYPED_TEST( ArrayViewTest, bind )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 );
   ViewType v;
   v.bind( a );
   EXPECT_EQ( v.getSize(), a.getSize() );
   EXPECT_EQ( v.getData(), a.getData() );

   // setting values
   a.setValue( 27 );
   EXPECT_EQ( a.getElement( 0 ), 27 );
   v.setValue( 50 );
   EXPECT_EQ( a.getElement( 0 ), 50 );
   a.reset();
   EXPECT_EQ( a.getSize(), 0 );
   EXPECT_EQ( v.getSize(), 10 );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ElementType data[ 10 ] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
      a.bind( data, 10 );
      EXPECT_EQ( a.getData(), data );
      EXPECT_EQ( a.getSize(), 10 );
      EXPECT_EQ( a.getElement( 1 ), 2 );
      v.bind( a );
      EXPECT_EQ( v.getElement( 1 ), 2 );
      a.reset();
      v.setElement( 1, 3 );
      v.reset();
      EXPECT_EQ( data[ 1 ], 3 );
   }
}

TYPED_TEST( ArrayViewTest, swap )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 ), b( 20 );
   a.setValue( 0 );
   b.setValue( 1 );

   ViewType u( a ), v( b );
   u.swap( v );
   EXPECT_EQ( u.getSize(), 20 );
   EXPECT_EQ( v.getSize(), 10 );
   for( int i = 0; i < 20; i++ )
      EXPECT_EQ( u.getElement( i ), 1 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
}

TYPED_TEST( ArrayViewTest, reset )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a;
   a.setSize( 100 );
   ViewType u( a );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
   u.bind( a );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
}

template< typename Element, typename Index >
void testArrayViewElementwiseAccess( Array< Element, Devices::Host, Index >&& a )
{
   a.setSize( 10 );
   using ViewType = ArrayView< Element, Devices::Host, Index >;
   ViewType u( a );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
   }
}

#ifdef HAVE_CUDA
template< typename ElementType, typename IndexType >
__global__ void testSetGetElementKernel( ArrayView< ElementType, Devices::Cuda, IndexType > v )
{
   if( threadIdx.x < v.getSize() )
      v[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

template< typename Element, typename Index >
void testArrayViewElementwiseAccess( Array< Element, Devices::Cuda, Index >&& u )
{
#ifdef HAVE_CUDA
   u.setSize( 10 );
   using ArrayType = Array< Element, Devices::Cuda, Index >;
   using ViewType = ArrayView< Element, Devices::Cuda, Index >;
   ViewType v( u );
   testSetGetElementKernel<<< 1, 16 >>>( v );
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( u.getElement( i ), i );
   }
#endif
}

template< typename Element, typename Index >
void testArrayViewElementwiseAccess( Array< Element, Devices::MIC, Index >&& u )
{
#ifdef HAVE_MIC
   // TODO
#endif
}

TYPED_TEST( ArrayViewTest, elementwiseAccess )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayViewElementwiseAccess( ArrayType() );
}

TYPED_TEST( ArrayViewTest, containsValue )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a;
   a.setSize( 1024 );
   ViewType v( a );

   for( int i = 0; i < v.getSize(); i++ )
      v.setElement( i, i % 10 );

   for( int i = 0; i < 10; i++ )
      EXPECT_TRUE( v.containsValue( i ) );

   for( int i = 10; i < 20; i++ )
      EXPECT_FALSE( v.containsValue( i ) );
}

TYPED_TEST( ArrayViewTest, containsOnlyValue )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a;
   a.setSize( 1024 );
   ViewType v( a );

   for( int i = 0; i < v.getSize(); i++ )
      v.setElement( i, i % 10 );

   for( int i = 0; i < 20; i++ )
      EXPECT_FALSE( v.containsOnlyValue( i ) );

   a.setValue( 100 );
   EXPECT_TRUE( v.containsOnlyValue( 100 ) );
}

TYPED_TEST( ArrayViewTest, comparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 ), b( 10 );
   typename ArrayType::HostType a_host( 10 );
   for( int i = 0; i < 10; i ++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
      b.setElement( i, 2 * i );
   }

   ViewType u( a ), v( a ), w( b );

   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   // comparison with arrays
   EXPECT_TRUE( a == u );
   EXPECT_FALSE( a != u );
   EXPECT_TRUE( u == a );
   EXPECT_FALSE( u != a );
   EXPECT_TRUE( a != w );
   EXPECT_FALSE( a == w );

   // comparison with different device
   EXPECT_TRUE( u == a_host );
   EXPECT_TRUE( a_host == u );
   // FIXME: what operator is called without explicit retyping?
//   EXPECT_TRUE( w != a_host );
   EXPECT_TRUE( w != (ArrayView< typename ArrayType::ElementType, Devices::Host, typename ArrayType::IndexType >) a_host );
   EXPECT_TRUE( a_host != w );

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

TYPED_TEST( ArrayViewTest, comparisonOperatorWithDifferentType )
{
   using DeviceType = typename TestFixture::ArrayType::DeviceType;
   using ArrayType1 = Array< short, DeviceType >;
   using ArrayType2 = Array< float, Devices::Host >;
   using ViewType1 = ArrayView< short, DeviceType >;
   using ViewType2 = ArrayView< float, Devices::Host >;

   ArrayType1 a( 10 );
   ArrayType2 b( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      b.setElement( i, i );
   }

   ViewType1 u( a );
   ViewType2 v( b );

   EXPECT_TRUE( u == v );
   EXPECT_FALSE( u != v );

   // the comparison will be in floats
   v.setElement( 0, 0.1f );
   EXPECT_FALSE( u == v );
   EXPECT_TRUE( u != v );
}

TYPED_TEST( ArrayViewTest, assignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 ), b( 10 );
   typename ArrayType::HostType a_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
   }

   ViewType u( a ), v( b );
   typename ViewType::HostType u_host( a_host );

   v.setValue( 0 );
   v = u;
   EXPECT_EQ( u, v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( u, v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from device to host
   u_host.setValue( 0 );
   u_host = u;
   EXPECT_EQ( u_host, u );
   EXPECT_EQ( u_host.getData(), a_host.getData() );
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ElementType >::value >::type >
void testArrayAssignmentWithDifferentType()
{
   ArrayType a( 10 );
   Array< short, typename ArrayType::DeviceType, short > b( 10 );
   Array< short, Devices::Host, short > b_host( 10 );
   typename ArrayType::HostType a_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
   }

   using ViewType = ArrayView< typename ArrayType::ElementType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;
   ViewType u( a );
   typename ViewType::HostType u_host( a_host );
   using ShortViewType = ArrayView< short, typename ArrayType::DeviceType, short >;
   ShortViewType v( b );
   typename ShortViewType::HostType v_host( b_host );

   v.setValue( 0 );
   v = u;
   EXPECT_EQ( v, u );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( v, u_host );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from device to host
   v_host.setValue( 0 );
   v_host = u;
   EXPECT_EQ( v_host, u );
   EXPECT_EQ( v_host.getData(), b_host.getData() );
}

template< typename ArrayType,
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ElementType >::value >::type,
          typename = void >
void testArrayAssignmentWithDifferentType()
{
}

TYPED_TEST( ArrayViewTest, assignmentOperatorWithDifferentType )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayAssignmentWithDifferentType< ArrayType >();
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
