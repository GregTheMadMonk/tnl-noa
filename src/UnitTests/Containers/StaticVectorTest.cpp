/***************************************************************************
                          StaticVectorTest.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_GTEST
#include <TNL/Containers/StaticVector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Vector >
class StaticVectorTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using RealType = typename VectorType::RealType;
};

// types for which VectorTest is instantiated
using StaticVectorTypes = ::testing::Types<
   StaticVector< 1, short >,
   StaticVector< 1, int >,
   StaticVector< 1, long >,
   StaticVector< 1, float >,
   StaticVector< 1, double >,
   StaticVector< 2, short >,
   StaticVector< 2, int >,
   StaticVector< 2, long >,
   StaticVector< 2, float >,
   StaticVector< 2, double >,
   StaticVector< 3, short >,
   StaticVector< 3, int >,
   StaticVector< 3, long >,
   StaticVector< 3, float >,
   StaticVector< 3, double >,
   StaticVector< 4, short >,
   StaticVector< 4, int >,
   StaticVector< 4, long >,
   StaticVector< 4, float >,
   StaticVector< 4, double >,
   StaticVector< 5, short >,
   StaticVector< 5, int >,
   StaticVector< 5, long >,
   StaticVector< 5, float >,
   StaticVector< 5, double >
>;

TYPED_TEST_CASE( StaticVectorTest, StaticVectorTypes );


TYPED_TEST( StaticVectorTest, constructors )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   constexpr int Size = VectorType::size;

   RealType data[ Size ];
   for( int i = 0; i < Size; i++ )
      data[ i ] = i;

   VectorType u0;
   EXPECT_TRUE( u0.getData() );

   VectorType u1( data );
   for( int i = 0; i < Size; i++ )
      EXPECT_EQ( u1[ i ], data[ i ] );

   VectorType u2( 7 );
   for( int i = 0; i < Size; i++ )
      EXPECT_EQ( u2[ i ], 7 );

   VectorType u3( u1 );
   for( int i = 0; i < Size; i++ )
      EXPECT_EQ( u3[ i ], u1[ i ] );

   // initialization with 0 requires special treatment to avoid ambiguity,
   // see https://stackoverflow.com/q/4610503
   VectorType v( 0 );
   for( int i = 0; i < Size; i++ )
      EXPECT_EQ( v[ i ], 0 );
}

TYPED_TEST( StaticVectorTest, operators )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   static_assert( Algorithms::Details::HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for StaticVector." );

   VectorType u1( 1 ), u2( 2 ), u3( 3 );

   u1 += u2;
   EXPECT_EQ( u1[ 0 ], 3 );
   EXPECT_EQ( u1[ size - 1 ], 3 );

   u1 -= u2;
   EXPECT_EQ( u1[ 0 ], 1 );
   EXPECT_EQ( u1[ size - 1 ], 1 );

   u1 *= 2;
   EXPECT_EQ( u1[ 0 ], 2 );
   EXPECT_EQ( u1[ size - 1 ], 2 );

   u3 = u1 + u2;
   EXPECT_EQ( u3[ 0 ], 4 );
   EXPECT_EQ( u3[ size - 1 ], 4 );

   u3 = u1 - u2;
   EXPECT_EQ( u3[ 0 ], 0 );
   EXPECT_EQ( u3[ size - 1 ], 0 );

   u3 = 2 * u1;
   EXPECT_EQ( u3[ 0 ], 4 );
   EXPECT_EQ( u3[ size - 1 ], 4 );

   EXPECT_EQ( ScalarProduct( u1, u2 ), 4 * size );
}

TYPED_TEST( StaticVectorTest, MinMax )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u1( 1 ), u2( 2 ), u3( 3 ), u4, u_min, u_max;
   for( int i = 0; i < size; i++ )
   {
      u4[ i ] = i;
      u_min[ i ] = TNL::min( i, 3 );
      u_max[ i ] = TNL::max( i, 3 );
   }

   EXPECT_TRUE( min( u1, u2 ) ==  u1 );
   EXPECT_TRUE( max( u1, u2 ) ==  u2 );
   EXPECT_TRUE( min( u3, u4 ) == u_min );
   EXPECT_TRUE( max( u3, u4 ) == u_max );
}

TYPED_TEST( StaticVectorTest, comparisons )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u1( 1 ), u2( 2 ), u3( 3 ), u4;
   for( int i = 0; i < size; i++ )
      u4[ i ] = i;

   EXPECT_TRUE( u1 < u3 );
   EXPECT_TRUE( u1 <= u3 );
   EXPECT_TRUE( u1 < u2 );
   EXPECT_TRUE( u1 <= u2 );
   EXPECT_TRUE( u3 > u1 );
   EXPECT_TRUE( u3 >= u1 );
   EXPECT_TRUE( u2 > u1 );
   EXPECT_TRUE( u2 >= u1 );
   EXPECT_TRUE( u1 != u4 );
   EXPECT_FALSE( u1 == u2 );

   if( size > 2 ) {
      EXPECT_FALSE( u1 < u4 );
      EXPECT_FALSE( u1 <= u4 );
      EXPECT_FALSE( u1 > u4 );
      EXPECT_FALSE( u1 >= u4 );
   }
}

TYPED_TEST( StaticVectorTest, cast )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u( 1 );
   EXPECT_EQ( (StaticVector< size, double >) u, u );
}

TYPED_TEST( StaticVectorTest, abs )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u;
   for( int i = 0; i < size; i++ )
      u[ i ] = i;

   VectorType v = -u;
   EXPECT_EQ( abs( v ), u );
}

TYPED_TEST( StaticVectorTest, sin )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = sin( u[ i ] );
   }

   EXPECT_EQ( sin( u ), v );
}

TYPED_TEST( StaticVectorTest, cos )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = cos( u[ i ] );
   }

   EXPECT_EQ( cos( u ), v );
}

TYPED_TEST( StaticVectorTest, tan )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = tan( u[ i ] );
   }

   EXPECT_EQ( tan( u ), v );
}

TYPED_TEST( StaticVectorTest, sqrt )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i;
      v[ i ] = sqrt( u[ i ] );
   }

   EXPECT_EQ( sqrt( u ), v );
}

TYPED_TEST( StaticVectorTest, cbrt )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i;
      v[ i ] = cbrt( u[ i ] );
   }

   EXPECT_EQ( cbrt( u ), v );
}

TYPED_TEST( StaticVectorTest, pow )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename VectorType::RealType;
   constexpr int size = VectorType::size;

   VectorType u, v, w;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = pow( u[ i ], 2.0 );
      w[ i ] = pow( u[ i ], 3.0 );
   }

   EXPECT_EQ( pow( u, 2.0 ), v );
   EXPECT_EQ( pow( u, 3.0 ), w );
}

TYPED_TEST( StaticVectorTest, floor )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = floor( u[ i ] );
   }

   EXPECT_EQ( floor( u ), v );
}

TYPED_TEST( StaticVectorTest, ceil )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = ceil( u[ i ] );
   }

   EXPECT_EQ( ceil( u ), v );
}

TYPED_TEST( StaticVectorTest, acos )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = ( double )( i - size / 2 ) / ( double ) size;
      v[ i ] = acos( u[ i ] );
   }

   EXPECT_EQ( acos( u ), v );
}

TYPED_TEST( StaticVectorTest, asin )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = ( double ) ( i - size / 2 ) / ( double ) size;
      v[ i ] = asin( u[ i ] );
   }

   EXPECT_EQ( asin( u ), v );
}

TYPED_TEST( StaticVectorTest, atan )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = atan( u[ i ] );
   }

   EXPECT_EQ( atan( u ), v );
}

TYPED_TEST( StaticVectorTest, cosh )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = cosh( u[ i ] );
   }

   // EXPECT_EQ( cosh( u ), v ) does not work here for float, maybe because
   // of some fast-math optimization
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( cosh( u )[ i ], v[ i ], 1.0e-6 );
}

TYPED_TEST( StaticVectorTest, tanh )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = tanh( u[ i ] );
   }

   EXPECT_EQ( tanh( u ), v );
}

TYPED_TEST( StaticVectorTest, log )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i + 1;
      v[ i ] = log( u[ i ] );
   }

   EXPECT_EQ( log( u ), v );
}

TYPED_TEST( StaticVectorTest, log10 )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i + 1;
      v[ i ] = log10( u[ i ] );
   }

   // EXPECT_EQ( log10( u ), v ) does not work here for float, maybe because
   // of some fast-math optimization
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( log10( u )[ i ], v[ i ], 1.0e-6 );
}

TYPED_TEST( StaticVectorTest, log2 )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i + 1;
      v[ i ] = log2( u[ i ] );
   }

   EXPECT_EQ( log2( u ), v );
}

TYPED_TEST( StaticVectorTest, exp )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = exp( u[ i ] );
   }

   EXPECT_EQ( exp( u ), v );
}

TYPED_TEST( StaticVectorTest, sign )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::size;

   VectorType u, v;
   for( int i = 0; i < size; i++ )
   {
      u[ i ] = i - size / 2;
      v[ i ] = sign( u[ i ] );
   }

   EXPECT_EQ( sign( u ), v );
}

TYPED_TEST( StaticVectorTest, lpNorm )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   constexpr int size = VectorType::size;
   const RealType epsilon = std::numeric_limits< RealType >::epsilon();

   VectorType v( 1 );

   const RealType expectedL1norm = size;
   const RealType expectedL2norm = std::sqrt( size );
   const RealType expectedL3norm = std::cbrt( size );
   EXPECT_EQ( v.lpNorm( 1.0 ), expectedL1norm );
   EXPECT_EQ( v.lpNorm( 2.0 ), expectedL2norm );
   EXPECT_NEAR( v.lpNorm( 3.0 ), expectedL3norm, epsilon );
}
#endif


#include "../main.h"
