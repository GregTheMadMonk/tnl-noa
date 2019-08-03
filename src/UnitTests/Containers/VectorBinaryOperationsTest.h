/***************************************************************************
                          VectorBinaryOperationsTest.h  -  description
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

template< typename T1, typename T2 >
struct Pair
{
   using Left = T1;
   using Right = T2;
};

// test fixture for typed tests
template< typename Pair >
class VectorBinaryOperationsTest : public ::testing::Test
{
protected:
   using Left = typename Pair::Left;
   using Right = typename Pair::Right;
   using LeftNonConstReal = std::remove_const_t< typename Left::RealType >;
   using RightNonConstReal = std::remove_const_t< typename Right::RealType >;
   using LeftVector = Vector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = Vector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
};

// types for which VectorBinaryOperationsTest is instantiated
using VectorPairs = ::testing::Types<
#ifndef HAVE_CUDA
   Pair< Vector<     int,       Devices::Host >, Vector<     int,       Devices::Host > >,
   Pair< VectorView< int,       Devices::Host >, Vector<     int,       Devices::Host > >,
   Pair< VectorView< const int, Devices::Host >, Vector<     int,       Devices::Host > >,
   Pair< Vector<     int,       Devices::Host >, VectorView< int,       Devices::Host > >,
   Pair< Vector<     int,       Devices::Host >, VectorView< const int, Devices::Host > >,
   Pair< VectorView< int,       Devices::Host >, VectorView< int,       Devices::Host > >,
   Pair< VectorView< const int, Devices::Host >, VectorView< int,       Devices::Host > >,
   Pair< VectorView< const int, Devices::Host >, VectorView< const int, Devices::Host > >,
   Pair< VectorView< int,       Devices::Host >, VectorView< const int, Devices::Host > >,
   Pair< Vector<     double,    Devices::Host >, Vector<     double,    Devices::Host > >,
   Pair< VectorView< double,    Devices::Host >, Vector<     double,    Devices::Host > >,
   Pair< Vector<     double,    Devices::Host >, VectorView< double,    Devices::Host > >,
   Pair< VectorView< double,    Devices::Host >, VectorView< double,    Devices::Host > >
#endif
#ifdef HAVE_CUDA
   Pair< Vector<     int,       Devices::Cuda >, Vector<     int,       Devices::Cuda > >,
   Pair< VectorView< int,       Devices::Cuda >, Vector<     int,       Devices::Cuda > >,
   Pair< VectorView< const int, Devices::Cuda >, Vector<     int,       Devices::Cuda > >,
   Pair< Vector<     int,       Devices::Cuda >, VectorView< int,       Devices::Cuda > >,
   Pair< Vector<     int,       Devices::Cuda >, VectorView< const int, Devices::Cuda > >,
   Pair< VectorView< int,       Devices::Cuda >, VectorView< int,       Devices::Cuda > >,
   Pair< VectorView< const int, Devices::Cuda >, VectorView< int,       Devices::Cuda > >,
   Pair< VectorView< const int, Devices::Cuda >, VectorView< const int, Devices::Cuda > >,
   Pair< VectorView< int,       Devices::Cuda >, VectorView< const int, Devices::Cuda > >,
   Pair< Vector<     double,    Devices::Cuda >, Vector<     double,    Devices::Cuda > >,
   Pair< VectorView< double,    Devices::Cuda >, Vector<     double,    Devices::Cuda > >,
   Pair< Vector<     double,    Devices::Cuda >, VectorView< double,    Devices::Cuda > >,
   Pair< VectorView< double,    Devices::Cuda >, VectorView< double,    Devices::Cuda > >
#endif
>;

TYPED_TEST_SUITE( VectorBinaryOperationsTest, VectorPairs );

#define SETUP_BINARY_VECTOR_TEST( size ) \
   using LeftVector = typename TestFixture::LeftVector;     \
   using RightVector = typename TestFixture::RightVector;   \
   using Left = typename TestFixture::Left;                 \
   using Right = typename TestFixture::Right;               \
                                                            \
   LeftVector _L1( size ), _L2( size );                     \
   RightVector _R1( size ), _R2( size );                    \
                                                            \
   _L1 = 1;                                                 \
   _L2 = 2;                                                 \
   _R1 = 1;                                                 \
   _R2 = 2;                                                 \
                                                            \
   Left L1( _L1 ), L2( _L2 );                               \
   Right R1( _R1 ), R2( _R2 );                              \

TYPED_TEST( VectorBinaryOperationsTest, EQ )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   EXPECT_EQ( L1, R1 );       // vector or vector view
   EXPECT_EQ( L1, 1 );        // right scalar
   EXPECT_EQ( 1, R1 );        // left scalar
   EXPECT_EQ( L2, R1 + R1 );  // right expression
   EXPECT_EQ( L1 + L1, R2 );  // left expression
   EXPECT_EQ( L1 + L1, R1 + R1 );  // two expressions

   // with different sizes
   EXPECT_FALSE( L1 == Right() );
   // with zero sizes
   EXPECT_TRUE( Left() == Right() );
}

TYPED_TEST( VectorBinaryOperationsTest, NE )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   EXPECT_NE( L1, R2 );       // vector or vector view
   EXPECT_NE( L1, 2 );        // right scalar
   EXPECT_NE( 2, R1 );        // left scalar
   EXPECT_NE( L1, R1 + R1 );  // right expression
   EXPECT_NE( L1 + L1, R1 );  // left expression
   EXPECT_NE( L1 + L1, R2 + R2 );  // two expressions

   // with different sizes
   EXPECT_TRUE( L1 != Right() );
   // with zero sizes
   EXPECT_FALSE( Left() != Right() );
}

TYPED_TEST( VectorBinaryOperationsTest, LT )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   EXPECT_LT( L1, R2 );       // vector or vector view
   EXPECT_LT( L1, 2 );        // right scalar
   EXPECT_LT( 1, R2 );        // left scalar
   EXPECT_LT( L1, R1 + R1 );  // right expression
   EXPECT_LT( L1 - L1, R1 );  // left expression
   EXPECT_LT( L1 - L1, R1 + R1 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, GT )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   EXPECT_GT( L2, R1 );       // vector or vector view
   EXPECT_GT( L2, 1 );        // right scalar
   EXPECT_GT( 2, R1 );        // left scalar
   EXPECT_GT( L1, R1 - R1 );  // right expression
   EXPECT_GT( L1 + L1, R1 );  // left expression
   EXPECT_GT( L1 + L1, R1 - R1 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, LE )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // same as LT
   EXPECT_LE( L1, R2 );       // vector or vector view
   EXPECT_LE( L1, 2 );        // right scalar
   EXPECT_LE( 1, R2 );        // left scalar
   EXPECT_LE( L1, R1 + R1 );  // right expression
   EXPECT_LE( L1 - L1, R1 );  // left expression
   EXPECT_LE( L1 - L1, R1 + R1 );  // two expressions

   // same as EQ
   EXPECT_LE( L1, R1 );       // vector or vector view
   EXPECT_LE( L1, 1 );        // right scalar
   EXPECT_LE( 1, R1 );        // left scalar
   EXPECT_LE( L2, R1 + R1 );  // right expression
   EXPECT_LE( L1 + L1, R2 );  // left expression
   EXPECT_LE( L1 + L1, R1 + R2 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, GE )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // same as GT
   EXPECT_GE( L2, R1 );       // vector or vector view
   EXPECT_GE( L2, 1 );        // right scalar
   EXPECT_GE( 2, R1 );        // left scalar
   EXPECT_GE( L1, R1 - R1 );  // right expression
   EXPECT_GE( L1 + L1, R1 );  // left expression
   EXPECT_GE( L1 + L1, R1 - R1 );  // two expressions

   // same as EQ
   EXPECT_LE( L1, R1 );       // vector or vector view
   EXPECT_LE( L1, 1 );        // right scalar
   EXPECT_LE( 1, R1 );        // left scalar
   EXPECT_LE( L2, R1 + R1 );  // right expression
   EXPECT_LE( L1 + L1, R2 );  // left expression
   EXPECT_LE( L1 + L1, R1 + R2 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, addition )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // with vector or vector view
   EXPECT_EQ( L1 + R1, 2 );
   // with scalar
   EXPECT_EQ( L1 + 1, 2 );
   EXPECT_EQ( 1 + L1, 2 );
   // with expression
   EXPECT_EQ( L1 + (L1 + L1), 3 );
   EXPECT_EQ( (L1 + L1) + L1, 3 );
   EXPECT_EQ( L1 + (L1 + R1), 3 );
   EXPECT_EQ( (L1 + L1) + R1, 3 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) + (L1 + L1), 4 );
}

TYPED_TEST( VectorBinaryOperationsTest, subtraction )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // with vector or vector view
   EXPECT_EQ( L1 - R1, 0 );
   // with scalar
   EXPECT_EQ( L1 - 1, 0 );
   EXPECT_EQ( 1 - L1, 0 );
   // with expression
   EXPECT_EQ( L2 - (L1 + L1), 0 );
   EXPECT_EQ( (L1 + L1) - L2, 0 );
   EXPECT_EQ( L2 - (L1 + R1), 0 );
   EXPECT_EQ( (L1 + L1) - R2, 0 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) - (L1 + L1), 0 );
}

TYPED_TEST( VectorBinaryOperationsTest, multiplication )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // with vector or vector view
   EXPECT_EQ( L1 * R2, L2 );
   // with scalar
   EXPECT_EQ( L1 * 2, L2 );
   EXPECT_EQ( 2 * L1, L2 );
   // with expression
   EXPECT_EQ( L1 * (L1 + L1), L2 );
   EXPECT_EQ( (L1 + L1) * L1, L2 );
   EXPECT_EQ( L1 * (L1 + R1), L2 );
   EXPECT_EQ( (L1 + L1) * R1, L2 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) * (L1 + L1), 4 );
}

TYPED_TEST( VectorBinaryOperationsTest, division )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // with vector or vector view
   EXPECT_EQ( L2 / R2, L1 );
   // with scalar
   EXPECT_EQ( L2 / 2, L1 );
   EXPECT_EQ( 2 / L2, L1 );
   // with expression
   EXPECT_EQ( L2 / (L1 + L1), L1 );
   EXPECT_EQ( (L1 + L1) / L2, L1 );
   EXPECT_EQ( L2 / (L1 + R1), L1 );
   EXPECT_EQ( (L1 + L1) / R2, L1 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) / (L1 + L1), L1 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   // with vector or vector view
   L1 = R2;
   EXPECT_EQ( L1, R2 );
   // with scalar
   L1 = 1;
   EXPECT_EQ( L1, 1 );
   // with expression
   L1 = R1 + R1;
   EXPECT_EQ( L1, R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, assignment )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );
   test_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_add_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_add_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   // with vector or vector view
   L1 += R2;
   EXPECT_EQ( L1, R1 + R2 );
   // with scalar
   L1 = 1;
   L1 += 2;
   EXPECT_EQ( L1, 3 );
   // with expression
   L1 = 1;
   L1 += R1 + R1;
   EXPECT_EQ( L1, R1 + R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, add_assignment )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );
   test_add_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_subtract_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_subtract_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   // with vector or vector view
   L1 -= R2;
   EXPECT_EQ( L1, R1 - R2 );
   // with scalar
   L1 = 1;
   L1 -= 2;
   EXPECT_EQ( L1, -1 );
   // with expression
   L1 = 1;
   L1 -= R1 + R1;
   EXPECT_EQ( L1, -R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, subtract_assignment )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );
   test_subtract_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_multiply_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_multiply_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   // with vector or vector view
   L1 *= R2;
   EXPECT_EQ( L1, R2 );
   // with scalar
   L1 = 1;
   L1 *= 2;
   EXPECT_EQ( L1, 2 );
   // with expression
   L1 = 1;
   L1 *= R1 + R1;
   EXPECT_EQ( L1, R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, multiply_assignment )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );
   test_multiply_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_divide_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_divide_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   // with vector or vector view
   L2 /= R2;
   EXPECT_EQ( L1, R1 );
   // with scalar
   L2 = 2;
   L2 /= 2;
   EXPECT_EQ( L1, 1 );
   // with expression
   L2 = 2;
   L2 /= R1 + R1;
   EXPECT_EQ( L1, R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, divide_assignment )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );
   test_divide_assignment( L1, L2, R1, R2 );
}

TYPED_TEST( VectorBinaryOperationsTest, scalarProduct )
{
   // this test expects an odd size
   const int size = VECTOR_TEST_REDUCTION_SIZE % 2 ? VECTOR_TEST_REDUCTION_SIZE : VECTOR_TEST_REDUCTION_SIZE - 1;

   using LeftVector = typename TestFixture::LeftVector;
   using RightVector = typename TestFixture::RightVector;
   using Left = typename TestFixture::Left;
   using Right = typename TestFixture::Right;

   LeftVector _L( size );
   RightVector _R( size );

   setOscilatingSequence( _L, 1 );
   setConstantSequence( _R, 1 );

   Left L( _L );
   Right R( _R );

   // vector or vector view
   EXPECT_EQ( dot(L, R), 1.0 );
   EXPECT_EQ( (L, R), 1.0 );
   // left expression
   EXPECT_EQ( dot(2 * L - L, R), 1.0 );
   EXPECT_EQ( (2 * L - L, R), 1.0 );
   // right expression
   EXPECT_EQ( dot(L, 2 * R - R), 1.0 );
   EXPECT_EQ( (L, 2 * R - R), 1.0 );
   // both expressions
   EXPECT_EQ( dot(2 * L - L, 2 * R - R), 1.0 );
   EXPECT_EQ( (2 * L - L, 2 * R - R), 1.0 );
}

TYPED_TEST( VectorBinaryOperationsTest, min )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

// FIXME: does not work without TNL::, because std::min conflicts (why?!)

   // with vector or vector view
   EXPECT_EQ( TNL::min(L1, R2), L1 );
   // with scalar
   EXPECT_EQ( TNL::min(L1, 2), L1 );
   EXPECT_EQ( TNL::min(1, R2), L1 );
   // with expression
   EXPECT_EQ( TNL::min(L1, R1 + R1), L1 );
   EXPECT_EQ( TNL::min(L1 + L1, R1), R1 );
   // with two expressions
   EXPECT_EQ( TNL::min(L1 + L1, R1 + R2), L2 );
}

TYPED_TEST( VectorBinaryOperationsTest, max )
{
   SETUP_BINARY_VECTOR_TEST( VECTOR_TEST_SIZE );

// FIXME: does not work without TNL::, because std::min conflicts (why?!)

   // with vector or vector view
   EXPECT_EQ( TNL::max(L1, R2), R2 );
   // with scalar
   EXPECT_EQ( TNL::max(L1, 2), L2 );
   EXPECT_EQ( TNL::max(1, R2), L2 );
   // with expression
   EXPECT_EQ( TNL::max(L1, R1 + R1), L2 );
   EXPECT_EQ( TNL::max(L1 + L1, R1), R2 );
   // with two expressions
   EXPECT_EQ( TNL::max(L1 - L1, R1 + R1), L2 );
}

//#ifdef HAVE_CUDA
//TYPED_TEST( VectorBinaryOperationsTest, comparisonOnDifferentDevices )
//{
//   using VectorType = typename TestFixture::VectorType;
//   const int size = VECTOR_TEST_SIZE;

//   typename VectorType::HostType host_vec( size );
//   typename VectorType::CudaType cuda_vec( size );
//   host_vec = 1.0;
//   cuda_vec = 1.0;
//   EXPECT_EQ( host_vec, cuda_vec );
//   EXPECT_EQ( host_vec.getView(), cuda_vec.getView() );

//   host_vec = 0.0;
//   EXPECT_TRUE( host_vec != cuda_vec );
//   EXPECT_TRUE( host_vec.getView() != cuda_vec.getView() );
//}
//#endif

#endif // HAVE_GTEST

#include "../main.h"
