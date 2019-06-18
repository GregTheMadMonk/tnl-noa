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
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/VectorView.h>

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
template< typename View >
class ArrayViewTest : public ::testing::Test
{
protected:
   using ViewType = View;
   using ArrayType = Array< typename View::ValueType, typename View::DeviceType, typename View::IndexType >;
};

// types for which ArrayViewTest is instantiated
using ViewTypes = ::testing::Types<
#ifndef HAVE_CUDA
   ArrayView< int,    Devices::Host, short >,
   ArrayView< long,   Devices::Host, short >,
   ArrayView< float,  Devices::Host, short >,
   ArrayView< double, Devices::Host, short >,
   ArrayView< MyData, Devices::Host, short >,
   ArrayView< int,    Devices::Host, int >,
   ArrayView< long,   Devices::Host, int >,
   ArrayView< float,  Devices::Host, int >,
   ArrayView< double, Devices::Host, int >,
   ArrayView< MyData, Devices::Host, int >,
   ArrayView< int,    Devices::Host, long >,
   ArrayView< long,   Devices::Host, long >,
   ArrayView< float,  Devices::Host, long >,
   ArrayView< double, Devices::Host, long >,
   ArrayView< MyData, Devices::Host, long >
   // FIXME: this segfaults in String::~String()
//   , ArrayView< String, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   ArrayView< int,    Devices::Cuda, short >,
   ArrayView< long,   Devices::Cuda, short >,
   ArrayView< float,  Devices::Cuda, short >,
   ArrayView< double, Devices::Cuda, short >,
   ArrayView< MyData, Devices::Cuda, short >,
   ArrayView< int,    Devices::Cuda, int >,
   ArrayView< long,   Devices::Cuda, int >,
   ArrayView< float,  Devices::Cuda, int >,
   ArrayView< double, Devices::Cuda, int >,
   ArrayView< MyData, Devices::Cuda, int >,
   ArrayView< int,    Devices::Cuda, long >,
   ArrayView< long,   Devices::Cuda, long >,
   ArrayView< float,  Devices::Cuda, long >,
   ArrayView< double, Devices::Cuda, long >,
   ArrayView< MyData, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   ArrayView< int,    Devices::MIC, short >,
   ArrayView< long,   Devices::MIC, short >,
   ArrayView< float,  Devices::MIC, short >,
   ArrayView< double, Devices::MIC, short >,
   // TODO: MyData does not work on MIC
//   ArrayView< MyData, Devices::MIC, short >,
   ArrayView< int,    Devices::MIC, int >,
   ArrayView< long,   Devices::MIC, int >,
   ArrayView< float,  Devices::MIC, int >,
   ArrayView< double, Devices::MIC, int >,
   // TODO: MyData does not work on MIC
//   ArrayView< MyData, Devices::MIC, int >,
   ArrayView< int,    Devices::MIC, long >,
   ArrayView< long,   Devices::MIC, long >,
   ArrayView< float,  Devices::MIC, long >,
   ArrayView< double, Devices::MIC, long >,
   // TODO: MyData does not work on MIC
//   ArrayView< MyData, Devices::MIC, long >,
#endif

   // all ArrayView tests should also work with VectorView
   // (but we can't test all types because the argument list would be too long...)
#ifndef HAVE_CUDA
   ,
   VectorView< float,  Devices::Host, long >,
   VectorView< double, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   ,
   VectorView< float,  Devices::Cuda, long >,
   VectorView< double, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   VectorView< float,  Devices::MIC, long >,
   VectorView< double, Devices::MIC, long >
#endif
>;

TYPED_TEST_SUITE( ArrayViewTest, ViewTypes );


TYPED_TEST( ArrayViewTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;

   ArrayType a( 10 );
   EXPECT_EQ( a.getSize(), 10 );

   ViewType v = a.getView();
   EXPECT_EQ( v.getSize(), 10 );
   EXPECT_EQ( v.getData(), a.getData() );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ValueType data[ 10 ];
      ViewType w( data, 10 );
      EXPECT_EQ( w.getData(), data );

      ViewType z( w );
      EXPECT_EQ( z.getData(), data );
      EXPECT_EQ( z.getSize(), 10 );
   }

   // test initialization by const reference
   const ArrayType& b = a;
   ConstViewType b_view = b.getConstView();
   EXPECT_EQ( b_view.getData(), b.getData() );
   ConstViewType const_a_view = a.getConstView();
   EXPECT_EQ( const_a_view.getData(), a.getData() );
   EXPECT_EQ( const_a_view.getSize(), a.getSize() );

   // test initialization of const view by non-const view
   ConstViewType const_b_view( b_view );
   EXPECT_EQ( const_b_view.getData(), b.getData() );
   EXPECT_EQ( const_b_view.getSize(), b.getSize() );
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

   ArrayType b = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   EXPECT_EQ( b.getSize(), 10 );
   EXPECT_EQ( b.getElement( 1 ), 2 );
   v.bind( b );
   EXPECT_EQ( v.getElement( 1 ), 2 );
   v.setElement( 1, 3 );
   EXPECT_EQ( b.getElement( 1 ), 3 );
}

TYPED_TEST( ArrayViewTest, swap )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 ), b( 20 );
   a.setValue( 0 );
   b.setValue( 1 );

   ViewType u = a.getView();
   ViewType v = b.getView();
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
   ViewType u = a.getView();
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

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::Host, Index >&& a )
{
   a.setSize( 10 );
   using ViewType = ArrayView< Value, Devices::Host, Index >;
   ViewType u( a );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
   }
}

#ifdef HAVE_CUDA
template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( ArrayView< ValueType, Devices::Cuda, IndexType > v )
{
   if( threadIdx.x < v.getSize() )
      v[ threadIdx.x ] = threadIdx.x;
}
#endif /* HAVE_CUDA */

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::Cuda, Index >&& u )
{
#ifdef HAVE_CUDA
   u.setSize( 10 );
   using ArrayType = Array< Value, Devices::Cuda, Index >;
   using ViewType = ArrayView< Value, Devices::Cuda, Index >;
   ViewType v( u );
   testSetGetElementKernel<<< 1, 16 >>>( v );
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( u.getElement( i ), i );
   }
#endif
}

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::MIC, Index >&& u )
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

template< typename ArrayType >
void ArrayViewEvaluateTest( ArrayType& u )
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using ViewType = ArrayView< ValueType, DeviceType, IndexType >;
   ViewType v( u );

   auto f = [] __cuda_callable__ ( IndexType i )
   {
      return 3 * i % 4;
   };

   v.evaluate( f );
   for( int i = 0; i < 10; i++ )
   {
      EXPECT_EQ( u.getElement( i ), 3 * i % 4 );
      EXPECT_EQ( v.getElement( i ), 3 * i % 4 );
   }
}

TYPED_TEST( ArrayViewTest, evaluate )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType u( 10 );
   ArrayViewEvaluateTest( u );
}

TYPED_TEST( ArrayViewTest, containsValue )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a;
   a.setSize( 1024 );
   ViewType v = a.getView();

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
   ViewType v = a.getView();

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

   ViewType u = a.getView();
   ViewType v = a.getView();
   ViewType w = b.getView();

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
   EXPECT_TRUE( w != (ArrayView< typename ArrayType::ValueType, Devices::Host, typename ArrayType::IndexType >) a_host );
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

   ViewType1 u = a.getView();
   ViewType2 v = b.getView();

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
   using ConstViewType = VectorView< const typename ArrayType::ValueType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;

   ArrayType a( 10 ), b( 10 );
   typename ArrayType::HostType a_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
   }

   ViewType u = a.getView();
   ViewType v = b.getView();
   typename ViewType::HostType u_host = a_host.getView();

   v.setValue( 0 );
   v = u;
   EXPECT_TRUE( u == v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from host to device
   //v.setValue( 0 );
   v = 0;
   v = u_host;
   // TODO: Replace with EXPECT_EQ when nvcc accepts it
   EXPECT_TRUE( u == v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from device to host
   /*u_host.setValue( 0 );
   u_host = u;
   
   EXPECT_TRUE( u_host == u );
   //EXPECT_EQ( u_host, u ); TODO: this is not accepted by nvcc 10, because nvcc is cockot
   EXPECT_EQ( u_host.getData(), a_host.getData() );

   // assignment of const view to non-const view
   v.setValue( 0 );
   ConstViewType c( u );
   v = c;*/
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ValueType >::value >::type >
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

   using ViewType = ArrayView< typename ArrayType::ValueType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;
   ViewType u = a.getView();
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
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ValueType >::value >::type,
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


#include "../main.h"
