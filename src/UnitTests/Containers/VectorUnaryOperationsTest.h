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

#if defined(DISTRIBUTED_VECTOR)
   #include <TNL/Communicators/MpiCommunicator.h>
   #include <TNL/Communicators/NoDistrCommunicator.h>
   #include <TNL/Containers/DistributedVector.h>
   #include <TNL/Containers/DistributedVectorView.h>
   #include <TNL/Containers/Partitioner.h>
#elif defined(STATIC_VECTOR)
   #include <TNL/Containers/StaticVector.h>
#else
   #include <TNL/Containers/Vector.h>
   #include <TNL/Containers/VectorView.h>
#endif

#include "VectorSequenceSetupFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

namespace unary_tests {

// prime number to force non-uniform distribution in block-wise algorithms
constexpr int VECTOR_TEST_SIZE = 97;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_REDUCTION_SIZE = 4999;

// test fixture for typed tests
template< typename T >
class VectorUnaryOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
#ifndef STATIC_VECTOR
   using NonConstReal = std::remove_const_t< typename VectorOrView::RealType >;
   #ifdef DISTRIBUTED_VECTOR
      using CommunicatorType = typename VectorOrView::CommunicatorType;
      using VectorType = DistributedVector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType, CommunicatorType >;
   #else
      using VectorType = Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   #endif
#endif
};

// types for which VectorUnaryOperationsTest is instantiated
#if defined(DISTRIBUTED_VECTOR)
   using VectorTypes = ::testing::Types<
   #ifndef HAVE_CUDA
      DistributedVector<           double, Devices::Host, int, Communicators::MpiCommunicator >,
      DistributedVectorView<       double, Devices::Host, int, Communicators::MpiCommunicator >,
      DistributedVectorView< const double, Devices::Host, int, Communicators::MpiCommunicator >,
      DistributedVector<           double, Devices::Host, int, Communicators::NoDistrCommunicator >,
      DistributedVectorView<       double, Devices::Host, int, Communicators::NoDistrCommunicator >,
      DistributedVectorView< const double, Devices::Host, int, Communicators::NoDistrCommunicator >
   #else
      DistributedVector<           double, Devices::Cuda, int, Communicators::MpiCommunicator >,
      DistributedVectorView<       double, Devices::Cuda, int, Communicators::MpiCommunicator >,
      DistributedVectorView< const double, Devices::Cuda, int, Communicators::MpiCommunicator >,
      DistributedVector<           double, Devices::Cuda, int, Communicators::NoDistrCommunicator >,
      DistributedVectorView<       double, Devices::Cuda, int, Communicators::NoDistrCommunicator >,
      DistributedVectorView< const double, Devices::Cuda, int, Communicators::NoDistrCommunicator >
   #endif
   >;
#elif defined(STATIC_VECTOR)
   using VectorTypes = ::testing::Types<
      StaticVector< 1, int >,
      StaticVector< 1, double >,
      StaticVector< 2, int >,
      StaticVector< 2, double >,
      StaticVector< 3, int >,
      StaticVector< 3, double >,
      StaticVector< 4, int >,
      StaticVector< 4, double >,
      StaticVector< 5, int >,
      StaticVector< 5, double >
   >;
#else
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
#endif

TYPED_TEST_SUITE( VectorUnaryOperationsTest, VectorTypes );

#ifdef STATIC_VECTOR
   #define SETUP_UNARY_VECTOR_TEST( _ ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = VectorOrView::getSize();            \
                                                               \
      VectorOrView V1, V2;                                     \
                                                               \
      V1 = 1;                                                  \
      V2 = 2;                                                  \

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _, begin, end, function ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorOrView::RealType;        \
      constexpr int size = VectorOrView::getSize();            \
                                                               \
      VectorOrView V1, expected;                               \
                                                               \
      const double h = (end - begin) / size;                   \
      for( int i = 0; i < size; i++ )                          \
      {                                                        \
         const RealType x = begin + i * h;                     \
         V1[ i ] = x;                                          \
         expected[ i ] = function(x);                          \
      }                                                        \

   #define SETUP_UNARY_VECTOR_TEST_REDUCTION \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = VectorOrView::getSize();            \
                                                               \
      VectorOrView V1;                                         \
      setLinearSequence( V1 );                                 \

#elif defined(DISTRIBUTED_VECTOR)
   #define SETUP_UNARY_VECTOR_TEST( _size ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = _size;                              \
      using CommunicatorType = typename VectorOrView::CommunicatorType; \
      const auto group = CommunicatorType::AllGroup; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType, CommunicatorType >::splitRange( size, group ); \
                                                               \
      const int rank = CommunicatorType::GetRank(group);       \
      const int nproc = CommunicatorType::GetSize(group);      \
                                                               \
      VectorType _V1, _V2;                                     \
      _V1.setDistribution( localRange, size, group );          \
      _V2.setDistribution( localRange, size, group );          \
                                                               \
      _V1 = 1;                                                 \
      _V2 = 2;                                                 \
                                                               \
      VectorOrView V1( _V1 ), V2( _V2 );                       \

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _size, begin, end, function ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorType::RealType;          \
      constexpr int size = _size;                              \
      using CommunicatorType = typename VectorOrView::CommunicatorType; \
      const auto group = CommunicatorType::AllGroup; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType, CommunicatorType >::splitRange( size, group ); \
                                                               \
      typename VectorType::HostType _V1h, expected_h;          \
      _V1h.setDistribution( localRange, size, group );         \
      expected_h.setDistribution( localRange, size, group );   \
                                                               \
      const double h = (end - begin) / size;                   \
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ ) \
      {                                                        \
         const RealType x = begin + i * h;                     \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      VectorType expected; expected = expected_h;              \

   #define SETUP_UNARY_VECTOR_TEST_REDUCTION \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = VECTOR_TEST_REDUCTION_SIZE;         \
      using CommunicatorType = typename VectorOrView::CommunicatorType; \
      const auto group = CommunicatorType::AllGroup; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType, CommunicatorType >::splitRange( size, group ); \
                                                               \
      VectorType _V1;                                          \
      _V1.setDistribution( localRange, size, group );          \
                                                               \
      setLinearSequence( _V1 );                                \
      VectorOrView V1( _V1 );                                  \

#else
   #define SETUP_UNARY_VECTOR_TEST( _size ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = _size;                              \
                                                               \
      VectorType _V1( size ), _V2( size );                     \
                                                               \
      _V1 = 1;                                                 \
      _V2 = 2;                                                 \
                                                               \
      VectorOrView V1( _V1 ), V2( _V2 );                       \

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _size, begin, end, function ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorType::RealType;          \
      constexpr int size = _size;                              \
                                                               \
      typename VectorType::HostType _V1h( size ), expected_h( size );  \
                                                               \
      const double h = (end - begin) / size;                   \
      for( int i = 0; i < size; i++ )                          \
      {                                                        \
         const RealType x = begin + i * h;                     \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      VectorType expected; expected = expected_h;              \

   #define SETUP_UNARY_VECTOR_TEST_REDUCTION \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      constexpr int size = VECTOR_TEST_REDUCTION_SIZE;         \
                                                               \
      VectorType _V1( size );                                  \
      setLinearSequence( _V1 );                                \
      VectorOrView V1( _V1 );                                  \

#endif

// This is because exact comparison does not work due to rounding errors:
// - the "expected" vector is computed sequentially on CPU
// - the host compiler might decide to use a vectorized version of the
//   math function, which may have slightly different precision
// - GPU may have different precision than CPU, so exact comparison with
//   the result from host is not possible
template< typename Left, typename Right >
void expect_vectors_near( const Left& _v1, const Right& _v2 )
{
   ASSERT_EQ( _v1.getSize(), _v2.getSize() );
#ifdef STATIC_VECTOR
   for( int i = 0; i < _v1.getSize(); i++ )
      EXPECT_NEAR( _v1[i], _v2[i], 1e-6 ) << "i = " << i;
#else
   using LeftNonConstReal = std::remove_const_t< typename Left::RealType >;
   using RightNonConstReal = std::remove_const_t< typename Right::RealType >;
#ifdef DISTRIBUTED_VECTOR
   using CommunicatorType = typename Left::CommunicatorType;
   static_assert( std::is_same< typename Right::CommunicatorType, CommunicatorType >::value,
                  "CommunicatorType must be the same for both Left and Right vectors." );
   using LeftVector = DistributedVector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType, CommunicatorType >;
   using RightVector = DistributedVector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType, CommunicatorType >;
#else
   using LeftVector = Vector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = Vector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
#endif
   using LeftHostVector = typename LeftVector::HostType;
   using RightHostVector = typename RightVector::HostType;

   // first evaluate expressions
   LeftVector v1; v1 = _v1;
   RightVector v2; v2 = _v2;
   // then copy to host
   LeftHostVector v1_h; v1_h = v1;
   RightHostVector v2_h; v2_h = v1;
#ifdef DISTRIBUTED_VECTOR
   const auto localRange = v1.getLocalRange();
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
#else
   for( int i = 0; i < v1.getSize(); i++ )
#endif
      EXPECT_NEAR( v1_h[i], v2_h[i], 1e-6 );
#endif
}

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
//   EXPECT_EQ( sin(V1), expected );
   expect_vectors_near( sin(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::asin );
//   EXPECT_EQ( asin(V1), expected );
   expect_vectors_near( asin(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cos );
//   EXPECT_EQ( cos(V1), expected );
   expect_vectors_near( cos(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::acos );
//   EXPECT_EQ( acos(V1), expected );
   expect_vectors_near( acos(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.5, 1.5, TNL::tan );
//   EXPECT_EQ( tan(V1), expected );
   expect_vectors_near( tan(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::atan );
//   EXPECT_EQ( atan(V1), expected );
   expect_vectors_near( atan(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::sinh );
//   EXPECT_EQ( sinh(V1), expected );
   expect_vectors_near( sinh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::asinh );
//   EXPECT_EQ( asinh(V1), expected );
   expect_vectors_near( asinh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::cosh );
//   EXPECT_EQ( cosh(V1), expected );
   expect_vectors_near( cosh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::acosh );
//   EXPECT_EQ( acosh(V1), expected );
   expect_vectors_near( acosh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::tanh );
//   EXPECT_EQ( tanh(V1), expected );
   expect_vectors_near( tanh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -0.99, 0.99, TNL::atanh );
//   EXPECT_EQ( atanh(V1), expected );
   expect_vectors_near( atanh(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, pow )
{
   auto pow3 = [](int i) { return TNL::pow(i, 3); };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, pow3 );
//   EXPECT_EQ( pow(V1, 3), expected );
   expect_vectors_near( pow(V1, 3), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, exp )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::exp );
//   EXPECT_EQ( exp(V1), expected );
   expect_vectors_near( exp(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sqrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 0, VECTOR_TEST_SIZE, TNL::sqrt );
//   EXPECT_EQ( sqrt(V1), expected );
   expect_vectors_near( sqrt(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cbrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cbrt );
//   EXPECT_EQ( cbrt(V1), expected );
   expect_vectors_near( cbrt(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log );
//   EXPECT_EQ( log(V1), expected );
   expect_vectors_near( log(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log10 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log10 );
//   EXPECT_EQ( log10(V1), expected );
   expect_vectors_near( log10(V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log2 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log2 );
//   EXPECT_EQ( log2(V1), expected );
   expect_vectors_near( log2(V1), expected );
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

TYPED_TEST( VectorUnaryOperationsTest, sign )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sign );
   EXPECT_EQ( sign(V1), expected );
}


TYPED_TEST( VectorUnaryOperationsTest, max )
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   EXPECT_EQ( max(V1), size - 1 );
   // unary expression
   EXPECT_EQ( max(-V1), 0 );
   // binary expression
   EXPECT_EQ( max(V1 + 2), size - 1 + 2 );
}

// FIXME: distributed argMax is not implemented yet
#ifdef DISTRIBUTED_VECTOR
TYPED_TEST( VectorUnaryOperationsTest, DISABLED_argMax )
#else
TYPED_TEST( VectorUnaryOperationsTest, argMax )
#endif
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   int arg = -1;
   EXPECT_EQ( argMax(V1, arg), size - 1 );
   EXPECT_EQ( arg, size - 1 );
   // unary expression
   arg = -1;
   EXPECT_EQ( argMax(-V1, arg), 0 );
   EXPECT_EQ( arg, 0 );
   // expression
   arg = -1;
   EXPECT_EQ( argMax(V1 + 2, arg), size - 1 + 2 );
   EXPECT_EQ( arg, size - 1 );
}

TYPED_TEST( VectorUnaryOperationsTest, min )
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   EXPECT_EQ( min(V1), 0 );
   // unary expression
   EXPECT_EQ( min(-V1), 1 - size );
   // binary expression
   EXPECT_EQ( min(V1 + 2), 2 );
}

// FIXME: distributed argMin is not implemented yet
#ifdef DISTRIBUTED_VECTOR
TYPED_TEST( VectorUnaryOperationsTest, DISABLED_argMin )
#else
TYPED_TEST( VectorUnaryOperationsTest, argMin )
#endif
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   int arg = -1;
   EXPECT_EQ( argMin(V1, arg), 0 );
   EXPECT_EQ( arg, 0 );
   // unary expression
   arg = -1;
   EXPECT_EQ( argMin(-V1, arg), 1 - size );
   EXPECT_EQ( arg, size - 1 );
   // binary expression
   arg = -1;
   EXPECT_EQ( argMin(V1 + 2, arg), 2 );
   EXPECT_EQ( arg, 0 );
}

TYPED_TEST( VectorUnaryOperationsTest, sum )
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   EXPECT_EQ( sum(V1), 0.5 * size * (size - 1) );
   // unary expression
   EXPECT_EQ( sum(-V1), - 0.5 * size * (size - 1) );
   // binary expression
   EXPECT_EQ( sum(V1 - 1), 0.5 * size * (size - 1) - size );
}

TYPED_TEST( VectorUnaryOperationsTest, maxNorm )
{
   SETUP_UNARY_VECTOR_TEST_REDUCTION;

   // vector or view
   EXPECT_EQ( maxNorm(V1), size - 1 );
   // unary expression
   EXPECT_EQ( maxNorm(-V1), size - 1 );
   // binary expression
   EXPECT_EQ( maxNorm(V1 - size), size );
}

TYPED_TEST( VectorUnaryOperationsTest, l1Norm )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_REDUCTION_SIZE );

   // vector or vector view
   EXPECT_EQ( l1Norm(V1), size );
   // unary expression
   EXPECT_EQ( l1Norm(-V1), size );
   // binary expression
   EXPECT_EQ( l1Norm(2 * V1 - V1), size );
}

TYPED_TEST( VectorUnaryOperationsTest, l2Norm )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_REDUCTION_SIZE );
   using RealType = typename VectorOrView::RealType;

   const RealType expected = std::sqrt( size );

   // vector or vector view
   EXPECT_EQ( l2Norm(V1), expected );
   // unary expression
   EXPECT_EQ( l2Norm(-V1), expected );
   // binary expression
   EXPECT_EQ( l2Norm(2 * V1 - V1), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, lpNorm )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_REDUCTION_SIZE );
   using RealType = typename VectorOrView::RealType;

   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   const RealType expectedL1norm = size;
   const RealType expectedL2norm = std::sqrt( size );
   const RealType expectedL3norm = std::cbrt( size );

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
   EXPECT_EQ( product(V2), std::exp2(size) );
   // unary expression
   EXPECT_EQ( product(-V2), std::exp2(size) * ( (size % 2) ? -1 : 1 ) );
   // binary expression
   EXPECT_EQ( product(V1 + V1), std::exp2(size) );
}

// TODO: tests for logicalOr, binaryOr, logicalAnd, binaryAnd

} // namespace unary_tests

#endif // HAVE_GTEST

#if !defined(DISTRIBUTED_VECTOR) && !defined(STATIC_VECTOR)
#include "../main.h"
#endif
