/***************************************************************************
                          ExpressionTemplatesTest.cpp  -  description
                             -------------------
    begin                : Mar 05, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by Vojtech Legler

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/Containers/StaticVector.h>

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_GTEST
TEST( ExpressionTemplatesStaticTest, Addition )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 9, 54, 300.4, 6 };
   StaticVector< 6, double > sv2{ 1.5, 1.5, 50, 30.4, 8, 600 };
   StaticVector< 6, double > svr1{};
   svr1 = sv1 + sv2 + sv2 + sv1;
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = sv1[ i ] + sv2[ i ] + sv2[ i ] + sv1[ i ];
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

/*TEST( ExpressionTemplatesStaticTest, Subtraction )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 9, 54, 300.4, 6 };
   StaticVector< 6, double > sv2{ 1.5, 1.5, 50, 30.4, 8, 600 };
   StaticVector< 6, double > svr1{};
   svr1 = sv2 - sv1 - sv1;
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = sv2[ i ] - sv1[ i ] - sv1[ i ];
   	EXPECT_EQ( svr1[ i ], temp );
   }
}


TEST( ExpressionTemplatesStaticTest, MultiplicationLeftSide )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 9, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = 5 * sv1;
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = 5 * sv1[ i ];
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, AbsoluteValue )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 9, 54, -300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = abs(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::abs( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, ExponentialFunction )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 0, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = exp(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::exp( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, NaturalLogarithm )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 1, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = log(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::log( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, Sine )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 0, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = sin(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::sin( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, Cosine )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 0, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = cos(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::cos( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, Tangent )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 0, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = tan(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::tan( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, ArcSine )
{
   StaticVector< 6, double > sv1{ 1, -0.5, 0, 0.35, -0.4, -1 };
   StaticVector< 6, double > svr1{};
   svr1 = asin(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::asin( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, ArcCosine )
{
   StaticVector< 6, double > sv1{ 1, -0.5, 0, 0.35, -0.4, -1 };
   StaticVector< 6, double > svr1{};
   svr1 = acos(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::acos( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesStaticTest, ArcTangent )
{
   StaticVector< 6, double > sv1{ 1, 1.5, 0, 54, 300.4, 6 };
   StaticVector< 6, double > svr1{};
   svr1 = atan(sv1);
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::atan( sv1[ i ] );
   	EXPECT_EQ( svr1[ i ], temp );
   }
}*/

#endif

#include "GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
