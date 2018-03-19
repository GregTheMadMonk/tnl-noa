/***************************************************************************
                          VectorTest.h  -  description
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

#include <TNL/Experimental/Arithmetics/MultiPrecision.h>
#include <TNL/Experimental/Arithmetics/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/File.h>
#include <TNL/Math.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics; 
using namespace TNL::Containers::Algorithms;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 5000;


template< typename Vector >
void setLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
   deviceVector = a;
}

template< typename Vector >
void setConstantSequence( Vector& deviceVector,
                          typename Vector::RealType v )
{
   deviceVector.setValue( v );
}

template< typename Vector >
void setNegativeLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
   deviceVector = a;
}

template< typename Vector >
void setOscilatingSequence( Vector& deviceVector,
                            typename Vector::RealType v )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   a[ 0 ] = v;
   for( int i = 1; i < a.getSize(); i++ )
      a[ i ] = a[ i-1 ] * -1;
   deviceVector = a;
}


// TODO: test everything with OpenMP with different number of threads

// test fixture for typed tests
template< typename Vector >
class VectorTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using VectorOperations = Algorithms::VectorOperations< typename VectorType::DeviceType >;
};

// types for which VectorTest is instantiated
using VectorTypes = ::testing::Types<
   Vector< int,    Devices::Host, short >,
   Vector< long,   Devices::Host, short >,
   Vector< float,  Devices::Host, short >,
   Vector< Quad<float>, Devices::Host, short>,
   Vector< double, Devices::Host, short >,
   Vector< Quad<double>, Devices::Host, short >,
   Vector< MultiPrecision, Devices::Host, short >,
   Vector< int,    Devices::Host, int >,
   Vector< long,   Devices::Host, int >,
   Vector< float,  Devices::Host, int >,
   Vector< Quad<float>, Devices::Host, int>,
   Vector< double, Devices::Host, int >,
   Vector< Quad<double>, Devices::Host, int>,
   Vector< MultiPrecision, Devices::Host, int >,
   Vector< int,    Devices::Host, long >,
   Vector< long,   Devices::Host, long >,
   Vector< float,  Devices::Host, long >,
   Vector< Quad<float>, Devices::Host, long>,
   Vector< double, Devices::Host, long >,
   Vector< Quad<double>, Devices::Host, long>,
   Vector< MultiPrecision, Devices::Host, long >
#ifdef HAVE_CUDA
   ,
   Vector< int,    Devices::Cuda, short >,
   Vector< long,   Devices::Cuda, short >,
   Vector< float,  Devices::Cuda, short >,
   Vector< Quad<float>, Devices::Cuda, short>,
   Vector< double, Devices::Cuda, short >,
   Vector< Quad<double>, Devices::Cuda, short>,
   Vector< MultiPrecision, Devices::Cuda, short>,
   Vector< int,    Devices::Cuda, int >,
   Vector< long,   Devices::Cuda, int >,
   Vector< float,  Devices::Cuda, int >,
   Vector< Quad<float>, Devices::Cuda, int>,
   Vector< double, Devices::Cuda, int >,
   Vector< Quad<double>, Devices::Cuda, int>,
   Vector< MultiPrecision, Devices::Cuda, int>,
   Vector< int,    Devices::Cuda, long >,
   Vector< long,   Devices::Cuda, long >,
   Vector< float,  Devices::Cuda, long >,
   Vector< Quad<float>, Devices::Cuda, long>,
   Vector< double, Devices::Cuda, long >,
   Vector< Quad<double>, Devices::Cuda, long>,
   Vector< MultiPrecision, Devices::Host, long >
#endif
#ifdef HAVE_MIC
   ,
   Vector< int,    Devices::MIC, short >,
   Vector< long,   Devices::MIC, short >,
   Vector< float,  Devices::MIC, short >,
   Vector< double, Devices::MIC, short >,
   Vector< int,    Devices::MIC, int >,
   Vector< long,   Devices::MIC, int >,
   Vector< float,  Devices::MIC, int >,
   Vector< double, Devices::MIC, int >,
   Vector< int,    Devices::MIC, long >,
   Vector< long,   Devices::MIC, long >,
   Vector< float,  Devices::MIC, long >,
   Vector< double, Devices::MIC, long >
#endif
>;

TYPED_TEST_CASE( VectorTest, VectorTypes );


TYPED_TEST( VectorTest, max )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   setLinearSequence( v );

   EXPECT_EQ( v.max(), size - 1 );
   EXPECT_EQ( VectorOperations::getVectorMax( v ), size - 1 );
}

TYPED_TEST( VectorTest, min )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   setLinearSequence( v );

   EXPECT_EQ( v.min(), 0 );
   EXPECT_EQ( VectorOperations::getVectorMin( v ), 0 );
}

TYPED_TEST( VectorTest, absMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   setNegativeLinearSequence( v );

   EXPECT_EQ( v.absMax(), size - 1 );
   EXPECT_EQ( VectorOperations::getVectorAbsMax( v ), size - 1 );
}

TYPED_TEST( VectorTest, absMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   setNegativeLinearSequence( v );

   EXPECT_EQ( v.absMin(), 0 );
   EXPECT_EQ( VectorOperations::getVectorAbsMin( v ), 0 );
}

TYPED_TEST( VectorTest, lpNorm )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename VectorType::RealType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;
   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   VectorType v;
   v.setSize( size );
   setConstantSequence( v, 1 );

   const RealType expectedL1norm = size;
   const RealType expectedL2norm = std::sqrt( size );
   const RealType expectedL3norm = std::cbrt( size );
   EXPECT_EQ( v.lpNorm( 1.0 ), expectedL1norm );
   EXPECT_EQ( v.lpNorm( 2.0 ), expectedL2norm );
   EXPECT_NEAR( v.lpNorm( 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( VectorOperations::getVectorLpNorm( v, 1.0 ), expectedL1norm );
   EXPECT_EQ( VectorOperations::getVectorLpNorm( v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( VectorOperations::getVectorLpNorm( v, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( VectorTest, sum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   // this test expect an even size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE - 1 : VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );

   setConstantSequence( v, 1 );
   EXPECT_EQ( v.sum(), size );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), size );

   setLinearSequence( v );
   EXPECT_EQ( v.sum(), 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), 0.5 * size * ( size - 1 ) );

   setNegativeLinearSequence( v );
   EXPECT_EQ( v.sum(), - 0.5 * size * ( size - 1 ) );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), - 0.5 * size * ( size - 1 ) );

   setOscilatingSequence( v, 1.0 );
   EXPECT_EQ( v.sum(), 0 );
   EXPECT_EQ( VectorOperations::getVectorSum( v ), 0 );
}

TYPED_TEST( VectorTest, differenceMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   setLinearSequence( u );
   setConstantSequence( v, size / 2 );

   EXPECT_EQ( u.differenceMax( v ), size - 1 - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMax( u, v ), size - 1 - size / 2 );
}

TYPED_TEST( VectorTest, differenceMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   setLinearSequence( u );
   setConstantSequence( v, size / 2 );

   EXPECT_EQ( u.differenceMin( v ), - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMin( u, v ), - size / 2 );
   EXPECT_EQ( v.differenceMin( u ), size / 2 - size + 1 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceMin( v, u ), size / 2 - size + 1 );
}

TYPED_TEST( VectorTest, differenceAbsMax )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   // this test expects an odd size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE : VECTOR_TEST_SIZE - 1;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   setNegativeLinearSequence( u );
   setConstantSequence( v, - size / 2 );

   EXPECT_EQ( u.differenceAbsMax( v ), size - 1 - size / 2 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMax( u, v ), size - 1 - size / 2 );
}

TYPED_TEST( VectorTest, differenceAbsMin )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   setNegativeLinearSequence( u );
   setConstantSequence( v, - size / 2 );

   EXPECT_EQ( u.differenceAbsMin( v ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( u, v ), 0 );
   EXPECT_EQ( v.differenceAbsMin( u ), 0 );
   EXPECT_EQ( VectorOperations::getVectorDifferenceAbsMin( v, u ), 0 );
}

TYPED_TEST( VectorTest, differenceLpNorm )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename VectorType::RealType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;
   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   u.setValue( 3.0 );
   v.setValue( 1.0 );

   const RealType expectedL1norm = 2.0 * size;
   const RealType expectedL2norm = std::sqrt( 4.0 * size );
   const RealType expectedL3norm = std::cbrt( 8.0 * size );
   EXPECT_EQ( u.differenceLpNorm( v, 1.0 ), expectedL1norm );
   EXPECT_EQ( u.differenceLpNorm( v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( u.differenceLpNorm( v, 3.0 ), expectedL3norm, epsilon );
   EXPECT_EQ( VectorOperations::getVectorDifferenceLpNorm( u, v, 1.0 ), expectedL1norm );
   EXPECT_EQ( VectorOperations::getVectorDifferenceLpNorm( u, v, 2.0 ), expectedL2norm );
   EXPECT_NEAR( VectorOperations::getVectorDifferenceLpNorm( u, v, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( VectorTest, differenceSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   // this test expect an even size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE - 1 : VECTOR_TEST_SIZE;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   v.setValue( 1.0 );

   setConstantSequence( u, 2 );
   EXPECT_EQ( u.differenceSum( v ), size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), size );

   setLinearSequence( u );
   EXPECT_EQ( u.differenceSum( v ), 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), 0.5 * size * ( size - 1 ) - size );

   setNegativeLinearSequence( u );
   EXPECT_EQ( u.differenceSum( v ), - 0.5 * size * ( size - 1 ) - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), - 0.5 * size * ( size - 1 ) - size );

   setOscilatingSequence( u, 1.0 );
   EXPECT_EQ( u.differenceSum( v ), - size );
   EXPECT_EQ( VectorOperations::getVectorDifferenceSum( u, v ), - size );
}

TYPED_TEST( VectorTest, scalarMultiplication )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType u;
   u.setSize( size );

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
   u *= 2.0;
   EXPECT_EQ( u, expected );
}

TYPED_TEST( VectorTest, scalarProduct )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   // this test expects an odd size
   const int size = VECTOR_TEST_SIZE % 2 ? VECTOR_TEST_SIZE : VECTOR_TEST_SIZE - 1;

   VectorType u, v;
   u.setSize( size );
   v.setSize( size );
   setOscilatingSequence( u, 1.0 );
   setConstantSequence( v, 1 );

   EXPECT_EQ( u.scalarProduct( v ), 1.0 );
   EXPECT_EQ( VectorOperations::getScalarProduct( u, v ), 1.0 );
}

TYPED_TEST( VectorTest, addVector )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y;
   x.setSize( size );
   y.setSize( size );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 2.0 + 3.0 * i;
      expected2[ i ] = 1.0 + 3.0 * i;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   VectorOperations::addVector( x, y, 3.0, 2.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   x.addVector( y, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );
}

TYPED_TEST( VectorTest, addVectors )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y, z;
   x.setSize( size );
   y.setSize( size );
   z.setSize( size );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 1.0 + 3.0 * i + 2.0;
      expected2[ i ] = 2.0 + 3.0 * i + 2.0;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   VectorOperations::addVectors( x, y, 3.0, z, 1.0, 1.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   x.addVectors( y, 3.0, z, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );
}

// TODO: fix the CUDA implementations
TYPED_TEST( VectorTest, prefixSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );

   setConstantSequence( v, 1 );
   v.computePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), i + 1 );

   v.setValue( 0 );
   v.computePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v.computePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i );
}

// TODO: fix the CUDA implementations
TYPED_TEST( VectorTest, exclusivePrefixSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );

   setConstantSequence( v, 1 );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), i );

   v.setValue( 0 );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v.computeExclusivePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i - 1 );
}

// TODO: test prefix sum with custom begin and end parameters

TEST( VectorSpecialCasesTest, sumOfBoolVector )
{
   using VectorType = Containers::Vector< bool, Devices::Host >;
   const float epsilon = 64 * std::numeric_limits< float >::epsilon();

   VectorType v( 512 ), w( 512 );
   v.setValue( true );
   w.setValue( false );

   const int sum = v.sum< int >();
   const int l1norm = v.lpNorm< int >( 1.0 );
   const float l2norm = v.lpNorm< float >( 2.0 );
   const float l3norm = v.lpNorm< float >( 3.0 );
   EXPECT_EQ( sum, 512 );
   EXPECT_EQ( l1norm, 512 );
   EXPECT_NEAR( l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm, std::cbrt( 512 ), epsilon );

   const int diff_sum = v.differenceSum< int >( w );
   const int diff_l1norm = v.differenceLpNorm< int >( w, 1.0 );
   const float diff_l2norm = v.differenceLpNorm< float >( w, 2.0 );
   const float diff_l3norm = v.differenceLpNorm< float >( w, 3.0 );
   EXPECT_EQ( diff_sum, 512 );
   EXPECT_EQ( diff_l1norm, 512 );
   EXPECT_NEAR( diff_l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm, std::cbrt( 512 ), epsilon );
}

#endif // HAVE_GTEST


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
