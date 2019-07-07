/***************************************************************************
                          VectorTest-3.h  -  description
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

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 5000;

TYPED_TEST( VectorTest, max )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   setLinearSequence( v );

   EXPECT_EQ( v.max(), size - 1 );
   EXPECT_EQ( v_view.max(), size - 1 );
   EXPECT_EQ( VectorOperations::getVectorMax( v ), size - 1 );
}

TYPED_TEST( VectorTest, min )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   setLinearSequence( v );

   EXPECT_EQ( v.min(), 0 );
   EXPECT_EQ( v_view.min(), 0 );
   EXPECT_EQ( VectorOperations::getVectorMin( v ), 0 );
}

TYPED_TEST( VectorTest, absMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   setNegativeLinearSequence( v );

   EXPECT_EQ( v.absMax(), size - 1 );
   EXPECT_EQ( v_view.absMax(), size - 1 );
   EXPECT_EQ( VectorOperations::getVectorAbsMax( v ), size - 1 );
}

TYPED_TEST( VectorTest, absMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   setNegativeLinearSequence( v );

   EXPECT_EQ( v.absMin(), 0 );
   EXPECT_EQ( v_view.absMin(), 0 );
   EXPECT_EQ( VectorOperations::getVectorAbsMin( v ), 0 );
}

TYPED_TEST( VectorTest, lpNorm )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename VectorType::RealType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;
   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   setConstantSequence( v, 1 );

   const RealType expectedL1norm = size;
   const RealType expectedL2norm = std::sqrt( size );
   const RealType expectedL3norm = std::cbrt( size );
   EXPECT_EQ( v.lpNorm( 1.0 ), expectedL1norm );
   EXPECT_EQ( v.lpNorm( 2.0 ), expectedL2norm );
   EXPECT_NEAR( v.lpNorm( 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( v_view.lpNorm( 1.0 ), expectedL1norm );
   EXPECT_EQ( v_view.lpNorm( 2.0 ), expectedL2norm );
   EXPECT_NEAR( v_view.lpNorm( 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( VectorOperations::getVectorLpNorm( v, 1.0 ), expectedL1norm );
   EXPECT_EQ( VectorOperations::getVectorLpNorm( v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( VectorOperations::getVectorLpNorm( v, 3.0 ), expectedL3norm, epsilon );
}

#endif // HAVE_GTEST

#include "../main.h"
