/***************************************************************************
                          VectorTest-5.h  -  description
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

   EXPECT_EQ( max( u - v ), size - 1 - size / 2 );
   EXPECT_EQ( max( u_view - v_view ), size - 1 - size / 2 );
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

   EXPECT_EQ( min( u - v ), - size / 2 );
   EXPECT_EQ( min( u_view - v_view ), - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMin( u, v ), - size / 2 );
   EXPECT_TRUE( min( v - u ) == size / 2 - size + 1 );
   EXPECT_TRUE( min( v_view - u_view ) == size / 2 - size + 1 );
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

   EXPECT_EQ( max( abs( u - v ) ), size - 1 - size / 2 );
   EXPECT_EQ( max( abs( u_view - v_view ) ), size - 1 - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMax( u, v ), size - 1 - size / 2 );
}

TYPED_TEST( VectorTest, differenceAbsMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType u( size ), v( size );
   ViewType u_view( u ), v_view( v );
   setNegativeLinearSequence( u );
   setConstantSequence( v, - size / 2 );

   EXPECT_EQ( min( abs( u - v ) ), 0 );
   EXPECT_EQ( min( abs( u_view - v_view ) ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( u, v ), 0 );
   EXPECT_EQ( min( abs( v - u ) ), 0 );
   EXPECT_EQ( min( abs( v_view - u_view ) ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( v, u ), 0 );
}

#endif // HAVE_GTEST

#include "../main.h"
