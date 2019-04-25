/***************************************************************************
                          ExpressionTemplatesTest.cpp  -  description
                             -------------------
    begin                : Mar 27, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by Vojtech Legler

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/Containers/Vector.h>
#include <TNL/Experimental/ExpressionTemplates/VectorExpressions.h>

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_GTEST
TEST( ExpressionTemplatesDynamicTest, Addition )
{  
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 9, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > d2{ 1.5, 1.5, 50, 30.4, 8, 600 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dv2( d2 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( dv1 + dv2 + dv2 + dv1 );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = dv1[ i ] + dv2[ i ] + dv2[ i ] + dv1[ i ];
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, Subtraction )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 9, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > d2{ 1.5, 1.5, 50, 30.4, 8, 600 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dv2( d2 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( dv2 - dv1 - dv1 );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = dv2[ i ] - dv1[ i ] - dv1[ i ];
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}


TEST( ExpressionTemplatesDynamicTest, MultiplicationLeftSide )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 9, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( 5*dv1 );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = 5 * dv1[ i ];
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, ExponentialFunction )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 0, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( TNL::exp(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::exp( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, NaturalLogarithm )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 1, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( TNL::log(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::log( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, Sine )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 0, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( sin(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::sin( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, Cosine )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 0, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( cos(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::cos( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, Tangent )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 0, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( tan(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::tan( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, ArcSine )
{
   Vector< double, Devices::Host, int > d1{ 1, -0.5, 0, 0.35, -0.4, -1 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( asin(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::asin( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, ArcCosine )
{
   Vector< double, Devices::Host, int > d1{ 1, -0.5, 0, 0.35, -0.4, -1 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( acos(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::acos( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

TEST( ExpressionTemplatesDynamicTest, ArcTangent )
{
   Vector< double, Devices::Host, int > d1{ 1, 1.5, 0, 54, 300.4, 6 };
   Vector< double, Devices::Host, int > dr1( 6 );
   VectorView< double, Devices::Host, int > dv1( d1 );
   VectorView< double, Devices::Host, int > dvr1( dr1 );
   dvr1.evaluate( atan(dv1) );
   double temp;
   for( int i = 0; i < 6; i++){
   	temp = std::atan( dv1[ i ] );
   	EXPECT_EQ( dvr1[ i ], temp );
   }
}

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
