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
   #ifdef VECTOR_OF_STATIC_VECTORS
      #include <TNL/Containers/StaticVector.h>
   #endif
   #include <TNL/Containers/Vector.h>
   #include <TNL/Containers/VectorView.h>
#endif

#include "VectorHelperFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

namespace unary_tests {

// prime number to force non-uniform distribution in block-wise algorithms
constexpr int VECTOR_TEST_SIZE = 97;

// test fixture for typed tests
template< typename T >
class VectorUnaryOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
#ifdef STATIC_VECTOR
   template< typename Real >
   using Vector = StaticVector< VectorOrView::getSize(), Real >;
#else
   using NonConstReal = std::remove_const_t< typename VectorOrView::RealType >;
   #ifdef DISTRIBUTED_VECTOR
      using CommunicatorType = typename VectorOrView::CommunicatorType;
      using VectorType = DistributedVector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType, CommunicatorType >;
      template< typename Real >
      using Vector = DistributedVector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType, CommunicatorType >;
   #else
      using VectorType = Containers::Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
      template< typename Real >
      using Vector = Containers::Vector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
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
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorTypes = ::testing::Types<
         StaticVector< 1, StaticVector< 3, double > >,
         StaticVector< 2, StaticVector< 3, double > >,
         StaticVector< 3, StaticVector< 3, double > >,
         StaticVector< 4, StaticVector< 3, double > >,
         StaticVector< 5, StaticVector< 3, double > >
      >;
   #else
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
   #endif
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorTypes = ::testing::Types<
      #ifndef HAVE_CUDA
         Vector<     StaticVector< 3, double >, Devices::Host >,
         VectorView< StaticVector< 3, double >, Devices::Host >
      #else
         Vector<     StaticVector< 3, double >, Devices::Cuda >,
         VectorView< StaticVector< 3, double >, Devices::Cuda >
      #endif
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
#endif

TYPED_TEST_SUITE( VectorUnaryOperationsTest, VectorTypes );

#ifdef STATIC_VECTOR
   #define SETUP_UNARY_VECTOR_TEST( _ ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
                                                               \
      VectorOrView V1, V2;                                     \
                                                               \
      V1 = 1;                                                  \
      V2 = 2;                                                  \

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _, begin, end, function ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorOrView::RealType;        \
      using ExpectedVector = typename TestFixture::template Vector< decltype(function(RealType{})) >; \
      constexpr int _size = VectorOrView::getSize();            \
                                                               \
      VectorOrView V1;                                         \
      ExpectedVector expected;                                 \
                                                               \
      const double h = (double) (end - begin) / _size;         \
      for( int i = 0; i < _size; i++ )                         \
      {                                                        \
         const RealType x = begin + i * h;                     \
         V1[ i ] = x;                                          \
         expected[ i ] = function(x);                          \
      }                                                        \

#elif defined(DISTRIBUTED_VECTOR)
   #define SETUP_UNARY_VECTOR_TEST( size ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using CommunicatorType = typename VectorOrView::CommunicatorType; \
      const auto group = CommunicatorType::AllGroup; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType, CommunicatorType >::splitRange( size, group ); \
                                                               \
      VectorType _V1, _V2;                                     \
      _V1.setDistribution( localRange, size, group );          \
      _V2.setDistribution( localRange, size, group );          \
                                                               \
      _V1 = 1;                                                 \
      _V2 = 2;                                                 \
                                                               \
      VectorOrView V1( _V1 ), V2( _V2 );                       \

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorType::RealType;          \
      using ExpectedVector = typename TestFixture::template Vector< decltype(function(RealType{})) >; \
      using HostVector = typename VectorType::template Self< RealType, Devices::Host >; \
      using HostExpectedVector = typename ExpectedVector::template Self< decltype(function(RealType{})), Devices::Host >; \
      using CommunicatorType = typename VectorOrView::CommunicatorType; \
      const auto group = CommunicatorType::AllGroup; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType, CommunicatorType >::splitRange( size, group ); \
                                                               \
      HostVector _V1h;                                         \
      HostExpectedVector expected_h;                           \
      _V1h.setDistribution( localRange, size, group );         \
      expected_h.setDistribution( localRange, size, group );   \
                                                               \
      const double h = (double) (end - begin) / size;          \
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ ) \
      {                                                        \
         const RealType x = begin + i * h;                     \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      ExpectedVector expected; expected = expected_h;          \

#else
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
      using ExpectedVector = typename TestFixture::template Vector< decltype(function(RealType{})) >; \
      using HostVector = typename VectorType::template Self< RealType, Devices::Host >; \
      using HostExpectedVector = typename ExpectedVector::template Self< decltype(function(RealType{})), Devices::Host >; \
                                                               \
      HostVector _V1h( size );                                 \
      HostExpectedVector expected_h( size );                   \
                                                               \
      const double h = (double) (end - begin) / size;          \
      for( int i = 0; i < size; i++ )                          \
      {                                                        \
         const RealType x = begin + i * h;                     \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      ExpectedVector expected; expected = expected_h;          \

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
   using LeftHostVector = typename LeftVector::template Self< LeftNonConstReal, Devices::Sequential >;
   using RightHostVector = typename RightVector::template Self< RightNonConstReal, Devices::Sequential >;

   // first evaluate expressions
   LeftVector v1; v1 = _v1;
   RightVector v2; v2 = _v2;
   // then copy to host
   LeftHostVector v1_h; v1_h = v1;
   RightHostVector v2_h; v2_h = v2;
#ifdef DISTRIBUTED_VECTOR
   const auto localRange = v1.getLocalRange();
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
#else
   for( int i = 0; i < v1.getSize(); i++ )
#endif
      EXPECT_NEAR( v1_h[i], v2_h[i], 1e-6 ) << "i = " << i;
#endif
}

TYPED_TEST( VectorUnaryOperationsTest, minus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( -V1, -1 );
   // unary expression
   EXPECT_EQ( V2 * (-V1), -2 );
   // binary expression
   EXPECT_EQ( -(V1 + V1), -2 );
}

TYPED_TEST( VectorUnaryOperationsTest, abs )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( abs(V1), V1 );
   // unary expression
   EXPECT_EQ( abs(-V1), V1 );
   // binary expression
   EXPECT_EQ( abs(-V1-V1), V2 );
}

TYPED_TEST( VectorUnaryOperationsTest, sin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sin );

   // vector or view
   expect_vectors_near( sin(V1), expected );
   // binary expression
   expect_vectors_near( sin(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sin(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::asin );

   // vector or view
   expect_vectors_near( asin(V1), expected );
   // binary expression
   expect_vectors_near( asin(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( asin(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cos );

   // vector or view
   expect_vectors_near( cos(V1), expected );
   // binary expression
   expect_vectors_near( cos(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cos(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::acos );

   // vector or view
   expect_vectors_near( acos(V1), expected );
   // binary expression
   expect_vectors_near( acos(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( acos(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.5, 1.5, TNL::tan );

   // vector or view
   expect_vectors_near( tan(V1), expected );
   // binary expression
   expect_vectors_near( tan(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( tan(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::atan );

   // vector or view
   expect_vectors_near( atan(V1), expected );
   // binary expression
   expect_vectors_near( atan(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( atan(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::sinh );

   // vector or view
   expect_vectors_near( sinh(V1), expected );
   // binary expression
   expect_vectors_near( sinh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sinh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::asinh );

   // vector or view
   expect_vectors_near( asinh(V1), expected );
   // binary expression
   expect_vectors_near( asinh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( asinh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::cosh );

   // vector or view
   expect_vectors_near( cosh(V1), expected );
   // binary expression
   expect_vectors_near( cosh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cosh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::acosh );

   // vector or view
   expect_vectors_near( acosh(V1), expected );
   // binary expression
   expect_vectors_near( acosh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( acosh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::tanh );

   // vector or view
   expect_vectors_near( tanh(V1), expected );
   // binary expression
   expect_vectors_near( tanh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( tanh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -0.99, 0.99, TNL::atanh );

   // vector or view
   expect_vectors_near( atanh(V1), expected );
   // binary expression
   expect_vectors_near( atanh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( atanh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, pow )
{
   // FIXME: for integer exponent, the test fails with CUDA
//   auto pow3 = [](double i) { return TNL::pow(i, 3); };
   auto pow3 = [](double i) { return TNL::pow(i, 3.0); };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, pow3 );

   // vector or view
   expect_vectors_near( pow(V1, 3.0), expected );
   // binary expression
   expect_vectors_near( pow(2 * V1 - V1, 3.0), expected );
   // unary expression
   expect_vectors_near( pow(-(-V1), 3.0), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, exp )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::exp );

   // vector or view
   expect_vectors_near( exp(V1), expected );
   // binary expression
   expect_vectors_near( exp(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( exp(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sqrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 0, VECTOR_TEST_SIZE, TNL::sqrt );

   // vector or view
   expect_vectors_near( sqrt(V1), expected );
   // binary expression
   expect_vectors_near( sqrt(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sqrt(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cbrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cbrt );

   // vector or view
   expect_vectors_near( cbrt(V1), expected );
   // binary expression
   expect_vectors_near( cbrt(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cbrt(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log );

   // vector or view
   expect_vectors_near( log(V1), expected );
   // binary expression
   expect_vectors_near( log(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log10 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log10 );

   // vector or view
   expect_vectors_near( log10(V1), expected );
   // binary expression
   expect_vectors_near( log10(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log10(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log2 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log2 );

   // vector or view
   expect_vectors_near( log2(V1), expected );
   // binary expression
   expect_vectors_near( log2(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log2(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, floor )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::floor );

   // vector or view
   expect_vectors_near( floor(V1), expected );
   // binary expression
   expect_vectors_near( floor(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( floor(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, ceil )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::ceil );

   // vector or view
   expect_vectors_near( ceil(V1), expected );
   // binary expression
   expect_vectors_near( ceil(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( ceil(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sign )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sign );

   // vector or view
   expect_vectors_near( sign(V1), expected );
   // binary expression
   expect_vectors_near( sign(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sign(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cast )
{
   auto identity = [](int i) { return i; };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, identity );

   // vector or vector view
   auto expression1 = cast<bool>(V1);
   static_assert( std::is_same< typename decltype(expression1)::RealType, bool >::value,
                  "BUG: the cast function does not work for vector or vector view." );
   EXPECT_EQ( expression1, true );

   // binary expression
   auto expression2( cast<bool>(V1 + V1) );
   static_assert( std::is_same< typename decltype(expression2)::RealType, bool >::value,
                  "BUG: the cast function does not work for binary expression." );
   // FIXME: expression2 cannot be reused, because expression templates for StaticVector and DistributedVector contain references and the test would crash in Release
//   EXPECT_EQ( expression2, true );
   EXPECT_EQ( cast<bool>(V1 + V1), true );

   // unary expression
   auto expression3( cast<bool>(-V1) );
   static_assert( std::is_same< typename decltype(expression3)::RealType, bool >::value,
                  "BUG: the cast function does not work for unary expression." );
   // FIXME: expression2 cannot be reused, because expression templates for StaticVector and DistributedVector contain references and the test would crash in Release
//   EXPECT_EQ( expression3, true );
   EXPECT_EQ( cast<bool>(-V1), true );
}

} // namespace unary_tests

#endif // HAVE_GTEST

#if !defined(DISTRIBUTED_VECTOR) && !defined(STATIC_VECTOR)
#include "../main.h"
#endif
