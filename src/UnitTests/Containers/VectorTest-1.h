/***************************************************************************
                          VectorTest-1.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include "VectorTestSetup.h"

constexpr int VECTOR_TEST_SIZE = 100;

TYPED_TEST( VectorTest, constructors )
{
   using VectorType = typename TestFixture::VectorType;
   const int size = VECTOR_TEST_SIZE;

   VectorType empty_u;
   VectorType empty_v( empty_u );
   EXPECT_EQ( empty_u.getSize(), 0 );
   EXPECT_EQ( empty_v.getSize(), 0 );

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

   static_assert( HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for Vector." );
   static_assert( HasSubscriptOperator< ViewType >::value, "Subscript operator detection by SFINAE does not work for VectorView." );

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
   EXPECT_EQ( sum( v_view ), 0 );
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

   typename VectorType::HostType host_expected1( size ), host_expected2( size );
   for( int i = 0; i < size; i++ ) {
      host_expected1.setElement( i, 2.0 + 3.0 * i );
      host_expected2.setElement( i, 1.0 + 3.0 * i );
   }
   VectorType expected1, expected2;
   expected1 = host_expected1;
   expected2 = host_expected2;

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   EXPECT_TRUE( 2.0 * x + 3.0 * y == expected1 );
   VectorOperations::addVector( x, y, 3.0, 2.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   EXPECT_TRUE( x + 3.0 * y == expected2 );
   x.addVector( y, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   EXPECT_TRUE( x_view + 3.0 * y_view == expected2 );
   x_view.addVector( y_view, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   EXPECT_EQ( min( -1.5 * x + 2.5 * y ), 7 );
   EXPECT_EQ( max( -1.5 * x + 2.5 * y ), 7 );
   x.addVector( y, 2.5, -1.5 );
   EXPECT_EQ( min( x ), 7 );
   EXPECT_EQ( max( x ), 7 );

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

   typename VectorType::HostType host_expected1( size ), host_expected2( size );
   for( int i = 0; i < size; i++ ) {
      host_expected1.setElement( i, 1.0 + 3.0 * i + 2.0 );
      host_expected2.setElement( i, 2.0 + 3.0 * i + 2.0 );
   }
   VectorType expected1, expected2;
   expected1 = host_expected1;
   expected2 = host_expected2;

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   EXPECT_TRUE( 3.0 * y + z + x == expected1 );
   VectorOperations::addVectors( x, y, 3.0, z, 1.0, 1.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   EXPECT_TRUE( 3.0 * y + z + 2.0 * x == expected2 );
   x.addVectors( y, 3.0, z, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   EXPECT_TRUE( 3.0 * y_view + z_view + 2.0 * x_view == expected2 );
   x_view.addVectors( y_view, 3.0, z_view, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   setConstantSequence( z, 6 );
   EXPECT_EQ( min( 2.5 * y - 1.5 * z - 1.5 * x ), -2 );
   EXPECT_EQ( max( 2.5 * y - 1.5 * z - 1.5 * x ), -2 );
   x.addVectors( y, 2.5, z, -1.5, -1.5 );
   EXPECT_EQ( min( x ), -2 );
   EXPECT_EQ( max( x ), -2 );
}

TEST( VectorSpecialCasesTest, sumOfBoolVector )
{
   using VectorType = Containers::Vector< bool, Devices::Host >;
   using ViewType = VectorView< bool, Devices::Host >;
   const float epsilon = 64 * std::numeric_limits< float >::epsilon();

   VectorType v( 512 ), w( 512 );
   ViewType v_view( v ), w_view( w );
   v.setValue( true );
   w.setValue( false );

   const int sum = TNL::sum( v );
   const int l1norm = lpNorm( v, 1.0 );
   const float l2norm = lpNorm( v, 2.0 );
   const float l3norm = lpNorm( v, 3.0 );
   EXPECT_EQ( sum, 512 );
   EXPECT_EQ( l1norm, 512 );
   EXPECT_NEAR( l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm, std::cbrt( 512 ), epsilon );

   const int diff_sum = TNL::sum( v - w );
   const int diff_l1norm = lpNorm( v - w, 1.0 );
   const float diff_l2norm = lpNorm( v - w, 2.0 );
   const float diff_l3norm = lpNorm( v - w, 3.0 );
   EXPECT_EQ( diff_sum, 512 );
   EXPECT_EQ( diff_l1norm, 512 );
   EXPECT_NEAR( diff_l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm, std::cbrt( 512 ), epsilon );

   // test views
   const int sum_view = TNL::sum( v_view );
   const int l1norm_view = lpNorm( v_view, 1.0 );
   const float l2norm_view = lpNorm( v_view, 2.0 );
   const float l3norm_view = lpNorm( v_view, 3.0 );
   EXPECT_EQ( sum_view, 512 );
   EXPECT_EQ( l1norm_view, 512 );
   EXPECT_NEAR( l2norm_view, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm_view, std::cbrt( 512 ), epsilon );

   const int diff_sum_view = TNL::sum( v_view - w_view );
   const int diff_l1norm_view = lpNorm( v_view -w_view, 1.0 );
   const float diff_l2norm_view = lpNorm( v_view - w_view, 2.0 );
   const float diff_l3norm_view = lpNorm( v_view - w_view, 3.0 );
   EXPECT_EQ( diff_sum_view, 512 );
   EXPECT_EQ( diff_l1norm_view, 512 );
   EXPECT_NEAR( diff_l2norm_view, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm_view, std::cbrt( 512 ), epsilon );
}

#endif // HAVE_GTEST

#include "../main.h"
