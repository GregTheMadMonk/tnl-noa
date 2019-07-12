/***************************************************************************
                          VectorTest-4.h  -  description
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

// Should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction.
constexpr int VECTOR_TEST_SIZE = 5000;

TYPED_TEST( VectorTest, sum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   // this test expect an even size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE - 1 : VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );

   setConstantSequence( v, 1 );
   EXPECT_EQ( sum( v ), size );
   EXPECT_EQ( sum( v_view ), size );

   setLinearSequence( v );
   EXPECT_EQ( sum( v ), 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( sum( v_view ), 0.5 * size * ( size - 1 ) );

   setNegativeLinearSequence( v );
   EXPECT_EQ( sum( v ), - 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( sum( v_view ), - 0.5 * size * ( size - 1 ) );

   setOscilatingSequence( v, 1.0 );
   EXPECT_EQ( sum( v ), 0 );
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

TYPED_TEST( VectorTest, scalarProduct )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   // this test expects an odd size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE : VECTOR_TEST_SIZE - 1;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   setOscilatingSequence( u, 1.0 );
   setConstantSequence( v, 1 );

   EXPECT_EQ( dot( u, v ), 1.0 );
   EXPECT_EQ( dot( u_view, v_view ), 1.0 );
   EXPECT_EQ( ( u, v ), 1.0 );
   EXPECT_EQ( ( u_view, v_view ), 1.0 );
}

TYPED_TEST( VectorTest, differenceLpNorm )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename VectorType::RealType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;
   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   u.setValue( 3.0 );
   v.setValue( 1.0 );

   const RealType expectedL1norm = 2.0 * size;
   const RealType expectedL2norm = std::sqrt( 4.0 * size );
   const RealType expectedL3norm = std::cbrt( 8.0 * size );
   EXPECT_EQ( lpNorm( u - v, 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( u - v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( u - v, 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( lpNorm( u_view - v_view, 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( u_view - v_view, 2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( u_view - v_view, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( VectorTest, differenceSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   // this test expect an even size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE - 1 : VECTOR_TEST_SIZE;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   v.setValue( 1.0 );

   setConstantSequence( u, 2 );
   EXPECT_EQ( sum( u - v ), size );
   EXPECT_EQ( sum( u_view - v_view ), size );

   setLinearSequence( u );
   EXPECT_EQ( sum( u - v ), 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( sum( u_view - v_view ), 0.5 * size * ( size - 1 ) - size );

   setNegativeLinearSequence( u );
   EXPECT_EQ( sum( u - v ), - 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( sum( u_view - v_view ), - 0.5 * size * ( size - 1 ) - size );

   setOscilatingSequence( u, 1.0 );
   EXPECT_EQ( sum( u - v ), - size );
   EXPECT_EQ( sum( u_view - v_view ), - size );
}

#endif // HAVE_GTEST

#include "../main.h"
