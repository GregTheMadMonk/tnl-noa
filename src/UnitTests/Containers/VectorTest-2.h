/***************************************************************************
                          VectorTest-2.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time

#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <TNL/Experimental/Arithmetics/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorTestSetup.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;
using namespace TNL::Arithmetics;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 5000;

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
   EXPECT_EQ( v.sum(), size );
   EXPECT_EQ( v_view.sum(), size );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), size );

   setLinearSequence( v );
   EXPECT_EQ( v.sum(), 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( v_view.sum(), 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), 0.5 * size * ( size - 1 ) );

   setNegativeLinearSequence( v );
   EXPECT_EQ( v.sum(), - 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( v_view.sum(), - 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), - 0.5 * size * ( size - 1 ) );

   setOscilatingSequence( v, 1.0 );
   EXPECT_EQ( v.sum(), 0 );
   EXPECT_EQ( v_view.sum(), 0 );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), 0 );
}

TYPED_TEST( VectorTest, differenceMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   setLinearSequence( u );
   setConstantSequence( v, size / 2 );

   EXPECT_EQ( u.differenceMax( v ), size - 1 - size / 2 );
   EXPECT_EQ( u_view.differenceMax( v_view ), size - 1 - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMax( u, v ), size - 1 - size / 2 );
}

TYPED_TEST( VectorTest, differenceMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   setLinearSequence( u );
   setConstantSequence( v, size / 2 );

   EXPECT_EQ( u.differenceMin( v ), - size / 2 );
   EXPECT_EQ( u_view.differenceMin( v_view ), - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMin( u, v ), - size / 2 );
   EXPECT_EQ( v.differenceMin( u ), size / 2 - size + 1 );
   EXPECT_EQ( v_view.differenceMin( u_view ), size / 2 - size + 1 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMin( v, u ), size / 2 - size + 1 );
}

TYPED_TEST( VectorTest, differenceAbsMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   // this test expects an odd size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE : VECTOR_TEST_SIZE - 1;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   setNegativeLinearSequence( u );
   setConstantSequence( v, - size / 2 );

   EXPECT_EQ( u.differenceAbsMax( v ), size - 1 - size / 2 );
   EXPECT_EQ( u_view.differenceAbsMax( v_view ), size - 1 - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMax( u, v ), size - 1 - size / 2 );
}


#endif // HAVE_GTEST

#include "../main.h"