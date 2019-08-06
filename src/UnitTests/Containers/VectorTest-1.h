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
