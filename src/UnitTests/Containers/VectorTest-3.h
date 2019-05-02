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

   EXPECT_EQ( u.differenceAbsMin( v ), 0 );
   EXPECT_EQ( u_view.differenceAbsMin( v_view ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( u, v ), 0 );
   EXPECT_EQ( v.differenceAbsMin( u ), 0 );
   EXPECT_EQ( v_view.differenceAbsMin( u_view ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( v, u ), 0 );
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
   EXPECT_EQ( u.differenceLpNorm( v, 1.0 ), expectedL1norm );
   EXPECT_EQ( u.differenceLpNorm( v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( u.differenceLpNorm( v, 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( u_view.differenceLpNorm( v_view, 1.0 ), expectedL1norm );
   EXPECT_EQ( u_view.differenceLpNorm( v_view, 2.0 ), expectedL2norm );
   EXPECT_NEAR( u_view.differenceLpNorm( v_view, 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( VectorOperations::getVectorDifferenceLpNorm( u, v, 1.0 ), expectedL1norm );
   EXPECT_EQ( VectorOperations::getVectorDifferenceLpNorm( u, v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( VectorOperations::getVectorDifferenceLpNorm( u, v, 3.0 ), expectedL3norm, epsilon );
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
   EXPECT_EQ( u.differenceSum( v ), size );
   EXPECT_EQ( u_view.differenceSum( v_view ), size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), size );

   setLinearSequence( u );
   EXPECT_EQ( u.differenceSum( v ), 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( u_view.differenceSum( v_view ), 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), 0.5 * size * ( size - 1 ) - size );

   setNegativeLinearSequence( u );
   EXPECT_EQ( u.differenceSum( v ), - 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( u_view.differenceSum( v_view ), - 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), - 0.5 * size * ( size - 1 ) - size );

   setOscilatingSequence( u, 1.0 );
   EXPECT_EQ( u.differenceSum( v ), - size );
   EXPECT_EQ( u_view.differenceSum( v_view ), - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), - size );
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

   EXPECT_EQ( u.scalarProduct( v ), 1.0 );
   EXPECT_EQ( u_view.scalarProduct( v_view ), 1.0 );
   EXPECT_EQ( VectorOperations::getScalarProduct( u, v ), 1.0 );
}

#endif // HAVE_GTEST


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
   //Test();
   //return 0;
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
