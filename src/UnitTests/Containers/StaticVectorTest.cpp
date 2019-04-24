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

   // TODO: implement unary minus operator
   VectorType v = - 1 * u;
   EXPECT_EQ( abs( v ), u );
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
