/***************************************************************************
                          VectorUnaryOperationsTest.h  -  description
                             -------------------
    begin                : Aug 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorSequenceSetupFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

constexpr int VECTOR_TEST_SIZE = 100;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_REDUCTION_SIZE = 5000;

// test fixture for typed tests
template< typename T >
class VectorUnaryOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
   using NonConstReal = std::remove_const_t< typename VectorOrView::RealType >;
   using VectorType = Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
};

// types for which VectorUnaryOperationsTest is instantiated
using VectorTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Vector<     int,       Devices::Host >,
   VectorView< int,       Devices::Host >,
   VectorView< const int, Devices::Host >,
   Vector<     double,    Devices::Host >,
   VectorView< double,    Devices::Host >
#endif
#ifdef HAVE_CUDA
   Vector<     int,       Devices::Cuda >,
   VectorView< int,       Devices::Cuda >,
   VectorView< const int, Devices::Cuda >,
   Vector<     double,    Devices::Cuda >,
   VectorView< double,    Devices::Cuda >
#endif
>;

TYPED_TEST_SUITE( VectorUnaryOperationsTest, VectorTypes );

#define SETUP_UNARY_VECTOR_TEST( size ) \
   using VectorType = typename TestFixture::VectorType;     \
   using VectorOrView = typename TestFixture::VectorOrView; \
                                                            \
   VectorType _V1( size ), _V2( size );                     \
                                                            \
   _V1 = 1;                                                 \
   _V2 = 2;                                                 \
                                                            \
   VectorOrView V1( _V1 ), V2( _V2 );                       \

#define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function ) \
   using VectorType = typename TestFixture::VectorType;     \
   using VectorOrView = typename TestFixture::VectorOrView; \
   using RealType = typename VectorType::RealType;          \
                                                            \
   typename VectorType::HostType _V1h( size ), expected( size );  \
                                                            \
   const double h = (end - begin) / size;                   \
   for( int i = 0; i < size; i++ )                          \
   {                                                        \
      const RealType x = begin + i * h;                     \
      _V1h[ i ] = x;                                        \
      expected[ i ] = function(x);                          \
   }                                                        \
                                                            \
   VectorType _V1; _V1 = _V1h;                              \
   VectorOrView V1( _V1 );                                  \

TYPED_TEST( VectorUnaryOperationsTest, minus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   EXPECT_EQ( -V1, -1 );
   EXPECT_EQ( V2 * (-V1), -2 );
   EXPECT_EQ( -(V1 + V1), -2 );
}

TYPED_TEST( VectorUnaryOperationsTest, abs )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( abs(V1), V1 );
   // expression
   EXPECT_EQ( abs(-V1), V1 );
}

TYPED_TEST( VectorUnaryOperationsTest, sin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sin );
   EXPECT_EQ( sin(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::asin );
   EXPECT_EQ( asin(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cos );
   EXPECT_EQ( cos(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::acos );
   EXPECT_EQ( acos(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.5, 1.5, TNL::tan );
   EXPECT_EQ( tan(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::atan );
   EXPECT_EQ( atan(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sqrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 0, VECTOR_TEST_SIZE, TNL::sqrt );
   EXPECT_EQ( sqrt(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cbrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cbrt );
   EXPECT_EQ( cbrt(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, pow )
{
   auto pow3 = [](int i) { return TNL::pow(i, 3); };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, pow3 );
   EXPECT_EQ( pow(V1, 3), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, floor )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::floor );
   EXPECT_EQ( floor(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, ceil )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::ceil );
   EXPECT_EQ( ceil(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sinh );
   EXPECT_EQ( sinh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::asinh );
   EXPECT_EQ( asinh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cosh );
   EXPECT_EQ( cosh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::acosh );
   EXPECT_EQ( acosh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::tanh );
   EXPECT_EQ( tanh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -0.99, 0.99, TNL::atanh );
   EXPECT_EQ( atanh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log );
   EXPECT_EQ( log(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log10 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log10 );
   EXPECT_EQ( log10(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log2 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log2 );
   EXPECT_EQ( log2(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, exp )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::exp );
   EXPECT_EQ( exp(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sign )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sign );
   EXPECT_EQ( sign(V1), expected );
}


TYPED_TEST( VectorUnaryOperationsTest, max )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOrView = typename TestFixture::VectorOrView;

   VectorType _V1( VECTOR_TEST_REDUCTION_SIZE ), _V2( VECTOR_TEST_REDUCTION_SIZE );
   setLinearSequence( _V1 );
   setConstantSequence( _V2, 2 );
   VectorOrView V1( _V1 ), V2( _V2 );

   // vector or view
   EXPECT_EQ( max(V1), VECTOR_TEST_REDUCTION_SIZE - 1 );
   // unary expression
   EXPECT_EQ( max(-V1), 0 );
   // binary expression
   EXPECT_EQ( max(V1 + V2), VECTOR_TEST_REDUCTION_SIZE - 1 + 2 );
}

TYPED_TEST( VectorUnaryOperationsTest, argMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOrView = typename TestFixture::VectorOrView;

   VectorType _V1( VECTOR_TEST_REDUCTION_SIZE ), _V2( VECTOR_TEST_REDUCTION_SIZE );
   setLinearSequence( _V1 );
   setConstantSequence( _V2, 2 );
   VectorOrView V1( _V1 ), V2( _V2 );

   // vector or view
   int arg = -1;
   EXPECT_EQ( argMax(V1, arg), VECTOR_TEST_REDUCTION_SIZE - 1 );
   EXPECT_EQ( arg, VECTOR_TEST_REDUCTION_SIZE - 1 );
   // unary expression
   arg = -1;
   EXPECT_EQ( argMax(-V1, arg), 0 );
   EXPECT_EQ( arg, 0 );
   // expression
   arg = -1;
   EXPECT_EQ( argMax(V1 + V2, arg), VECTOR_TEST_REDUCTION_SIZE - 1 + 2 );
   EXPECT_EQ( arg, VECTOR_TEST_REDUCTION_SIZE - 1 );
}

TYPED_TEST( VectorUnaryOperationsTest, min )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOrView = typename TestFixture::VectorOrView;

   VectorType _V1( VECTOR_TEST_REDUCTION_SIZE ), _V2( VECTOR_TEST_REDUCTION_SIZE );
   setLinearSequence( _V1 );
   setConstantSequence( _V2, 2 );
   VectorOrView V1( _V1 ), V2( _V2 );

   // vector or view
   EXPECT_EQ( min(V1), 0 );
   // unary expression
   EXPECT_EQ( min(-V1), 1 - VECTOR_TEST_REDUCTION_SIZE );
   // binary expression
   EXPECT_EQ( min(V1 + V2), 2 );
}

TYPED_TEST( VectorUnaryOperationsTest, argMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOrView = typename TestFixture::VectorOrView;

   VectorType _V1( VECTOR_TEST_REDUCTION_SIZE ), _V2( VECTOR_TEST_REDUCTION_SIZE );
   setLinearSequence( _V1 );
   setConstantSequence( _V2, 2 );
   VectorOrView V1( _V1 ), V2( _V2 );

   // vector or view
   int arg = -1;
   EXPECT_EQ( argMin(V1, arg), 0 );
   EXPECT_EQ( arg, 0 );
   // unary expression
   arg = -1;
   EXPECT_EQ( argMin(-V1, arg), 1 - VECTOR_TEST_REDUCTION_SIZE );
   EXPECT_EQ( arg, VECTOR_TEST_REDUCTION_SIZE - 1 );
   // binary expression
   arg = -1;
   EXPECT_EQ( argMin(V1 + V2, arg), 2 );
   EXPECT_EQ( arg, 0 );
}

TYPED_TEST( VectorUnaryOperationsTest, sum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOrView = typename TestFixture::VectorOrView;
   // this test expect an even size
   const int size = VECTOR_TEST_REDUCTION_SIZE % 2 ? VECTOR_TEST_REDUCTION_SIZE - 1 : VECTOR_TEST_REDUCTION_SIZE;

   VectorType _V1( size ), _V2( size );
   setLinearSequence( _V1 );
   setConstantSequence( _V2, 1 );
   VectorOrView V1( _V1 ), V2( _V2 );

   // vector or view
   EXPECT_EQ( sum(V1), 0.5 * size * (size - 1) );
   // unary expression
   EXPECT_EQ( sum(-V1), - 0.5 * size * (size - 1) );
   // binary expression
   EXPECT_EQ( sum(V1 - V2), 0.5 * size * (size - 1) - size );
}

TYPED_TEST( VectorUnaryOperationsTest, lpNorm )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_REDUCTION_SIZE );
   using RealType = typename VectorOrView::RealType;

   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   const RealType expectedL1norm = VECTOR_TEST_REDUCTION_SIZE;
   const RealType expectedL2norm = std::sqrt( VECTOR_TEST_REDUCTION_SIZE );
   const RealType expectedL3norm = std::cbrt( VECTOR_TEST_REDUCTION_SIZE );

   // vector or vector view
   EXPECT_EQ( lpNorm(V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(V1, 2.0), expectedL2norm );
   EXPECT_NEAR( lpNorm(V1, 3.0), expectedL3norm, epsilon );
   // unary expression
   EXPECT_EQ( lpNorm(-V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(-V1, 2.0), expectedL2norm );
   EXPECT_NEAR( lpNorm(-V1, 3.0), expectedL3norm, epsilon );
   // binary expression
   EXPECT_EQ( lpNorm(2 * V1 - V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(2 * V1 - V1, 2.0), expectedL2norm );
   EXPECT_NEAR( lpNorm(2 * V1 - V1, 3.0), expectedL3norm, epsilon );
}

TYPED_TEST( VectorUnaryOperationsTest, product )
{
   SETUP_UNARY_VECTOR_TEST( 16 );

   // vector or vector view
   EXPECT_EQ( product(V2), std::exp2(16) );
   // unary expression
   EXPECT_EQ( product(-V2), std::exp2(16) );
   // binary expression
   EXPECT_EQ( product(V1 + V1), std::exp2(16) );
}

// TODO: tests for logicalOr, binaryOr, logicalAnd, binaryAnd

#endif // HAVE_GTEST

#include "../main.h"
