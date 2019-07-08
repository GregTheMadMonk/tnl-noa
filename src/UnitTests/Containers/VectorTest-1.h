/***************************************************************************
                          VectorTest-1.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time

#pragma once

#ifdef HAVE_GTEST
#include "VectorTestSetup.h"

constexpr int VECTOR_TEST_SIZE = 100;

TYPED_TEST( VectorTest, constructors )
{
   using VectorType = typename TestFixture::VectorType;
   const int size = VECTOR_TEST_SIZE;

   // TODO: Does not work yet.
   /*VectorType empty_u;
   VectorType empty_v( empty_u );
   EXPECT_EQ( empty_u.getSize(), 0 );
   EXPECT_EQ( empty_v.getSize(), 0 );*/

   VectorType u( size );
   EXPECT_EQ( u.getSize(), size );

   VectorType v( 10 );
   EXPECT_EQ( v.getSize(), 10 );

   if( std::is_same< typename VectorType::DeviceType, Devices::Host >::value ) {
      typename VectorType::ValueType data[ 10 ];
      VectorType w( data, 10 );
      EXPECT_NE( w.getData(), data );

      VectorType z1( w );
      EXPECT_NE( z1.getData(), data );
      EXPECT_EQ( z1.getSize(), 10 );

      VectorType z2( w, 1 );
      EXPECT_EQ( z2.getSize(), 9 );

      VectorType z3( w, 2, 3 );
      EXPECT_EQ( z3.getSize(), 3 );
   }

   v = 1;
   VectorType w( v );
   EXPECT_EQ( w.getSize(), v.getSize() );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), w.getElement( i ) );
   v.reset();
   EXPECT_EQ( w.getSize(), 10 );

   VectorType a1 { 1, 2, 3 };
   EXPECT_EQ( a1.getElement( 0 ), 1 );
   EXPECT_EQ( a1.getElement( 1 ), 2 );
   EXPECT_EQ( a1.getElement( 2 ), 3 );

   std::list< int > l = { 4, 5, 6 };
   VectorType a2( l );
   EXPECT_EQ( a2.getElement( 0 ), 4 );
   EXPECT_EQ( a2.getElement( 1 ), 5 );
   EXPECT_EQ( a2.getElement( 2 ), 6 );

   std::vector< int > q = { 7, 8, 9 };

   VectorType a3( q );
   EXPECT_EQ( a3.getElement( 0 ), 7 );
   EXPECT_EQ( a3.getElement( 1 ), 8 );
   EXPECT_EQ( a3.getElement( 2 ), 9 );
}

TEST( VectorSpecialCasesTest, defaultConstructors )
{
   using ArrayType = Containers::Array< int, Devices::Host >;
   using VectorViewType = VectorView< int, Devices::Host >;
   using ArrayViewType = ArrayView< int, Devices::Host >;

   ArrayType a( 100 );
   a.setValue( 0 );

   ArrayViewType a_view;
   a_view.bind( a );

   VectorViewType v_view;
   v_view.bind( a );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
}

TEST( VectorSpecialCasesTest, assignmentThroughView )
{
   using VectorType = Containers::Vector< int, Devices::Host >;
   using ViewType = VectorView< int, Devices::Host >;

   static_assert( Algorithms::detail::HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for Vector." );
   static_assert( Algorithms::detail::HasSubscriptOperator< ViewType >::value, "Subscript operator detection by SFINAE does not work for VectorView." );

   VectorType u( 100 ), v( 100 );
   ViewType u_view( u ), v_view( v );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u_view;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view[ i ], 42 );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view[ i ], 42 );
}

TEST( VectorSpecialCasesTest, operationsOnConstView )
{
   using VectorType = Containers::Vector< int, Devices::Host >;
   using ViewType = VectorView< const int, Devices::Host >;

   VectorType u( 100 ), v( 100 );
   ViewType u_view( u ), v_view( v );

   u.setValue( 1 );
   v.setValue( 1 );

   EXPECT_EQ( u_view.max(), 1 );
   EXPECT_EQ( u_view.min(), 1 );
   EXPECT_EQ( u_view.absMax(), 1 );
   EXPECT_EQ( u_view.absMin(), 1 );
   EXPECT_EQ( u_view.lpNorm( 1 ), 100 );
   EXPECT_EQ( u_view.differenceMax( v_view ), 0 );
   EXPECT_EQ( u_view.differenceMin( v_view ), 0 );
   EXPECT_EQ( u_view.differenceAbsMax( v_view ), 0 );
   EXPECT_EQ( u_view.differenceAbsMin( v_view ), 0 );
   EXPECT_EQ( u_view.differenceLpNorm( v_view, 1 ), 0 );
   EXPECT_EQ( u_view.differenceSum( v_view ), 0 );
   EXPECT_EQ( u_view.scalarProduct( v_view ), 100 );
}

TEST( VectorSpecialCasesTest, initializationOfVectorViewByArrayView )
{
   using ArrayType = Containers::Array< int, Devices::Host >;
   using VectorViewType = VectorView< const int, Devices::Host >;
   using ArrayViewType = ArrayView< int, Devices::Host >;

   ArrayType a( 100 );
   a.setValue( 0 );
   ArrayViewType a_view( a );

   VectorViewType v_view( a_view );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
   EXPECT_EQ( v_view.sum(), 0 );
}

TYPED_TEST( VectorTest, scalarMultiplication )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType u( size );
   ViewType u_view( u );

   typename VectorType::HostType expected;
   expected.setSize( size );
   for( int i = 0; i < size; i++ )
      expected[ i ] = 2.0 * i;

   setLinearSequence( u );
   VectorOperations::vectorScalarMultiplication( u, 2.0 );
   EXPECT_EQ( u, expected );

   setLinearSequence( u );
   u.scalarMultiplication( 2.0 );
   EXPECT_EQ( u, expected );

   setLinearSequence( u );
   u_view.scalarMultiplication( 2.0 );
   EXPECT_EQ( u, expected );

   setLinearSequence( u );
   u *= 2.0;
   EXPECT_EQ( u, expected );

   setLinearSequence( u );
   u_view *= 2.0;
   EXPECT_EQ( u, expected );
}

TYPED_TEST( VectorTest, addVector )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y;
   x.setSize( size );
   y.setSize( size );
   ViewType x_view( x ), y_view( y );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 2.0 + 3.0 * i;
      expected2[ i ] = 1.0 + 3.0 * i;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   VectorOperations::addVector( x, y, 3.0, 2.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   x.addVector( y, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   x_view.addVector( y_view, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   x.addVector( y, 2.5, -1.5 );
   EXPECT_EQ( x.min(), 7 );
   EXPECT_EQ( x.max(), 7 );
}

TYPED_TEST( VectorTest, addVectors )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y, z;
   x.setSize( size );
   y.setSize( size );
   z.setSize( size );
   ViewType x_view( x ), y_view( y ), z_view( z );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 1.0 + 3.0 * i + 2.0;
      expected2[ i ] = 2.0 + 3.0 * i + 2.0;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   VectorOperations::addVectors( x, y, 3.0, z, 1.0, 1.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   x.addVectors( y, 3.0, z, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   x_view.addVectors( y_view, 3.0, z_view, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   setConstantSequence( z, 6 );
   x.addVectors( y, 2.5, z, -1.5, -1.5 );
   EXPECT_EQ( x.min(), -2 );
   EXPECT_EQ( x.max(), -2 );
}

TYPED_TEST( VectorTest, abs )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
      u.setElement( i, i );

   v = -u;
   EXPECT_TRUE( abs( v ) == u );
}

TYPED_TEST( VectorTest, comparison )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _v( size ), _w( size );
   ViewType v( _v ), w( _w );
   v = 1.0;
   w = 2.0;

   EXPECT_TRUE( v < w );
   EXPECT_TRUE( w > v );
   EXPECT_TRUE( w + 1.0 < v + 4.0 );
}

TYPED_TEST( VectorTest, horizontalOperations )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   EXPECT_EQ( u.getSize(), size );
   u = 0;
   v = 1;
   w = 2;

   u = u + 4 * TNL::max( v, 0 );
   EXPECT_TRUE( u.containsOnlyValue( 4.0 ) );

   u = u + 3 * w + 4 * TNL::max( v, 0 );
   EXPECT_TRUE( u.containsOnlyValue( 14.0 ) );
}

#endif // HAVE_GTEST

#include "../main.h"
