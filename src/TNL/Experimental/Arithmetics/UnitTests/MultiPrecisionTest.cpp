/**************************************************
* filename:		MultiPrecisionTest.cpp	  *
* created:		December 1, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#ifdef HAVE_GMP
#include <gmp.h>
#endif

#include <TNL/Experimental/Arithmetics/MultiPrecision.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( MultiPrecisionTest, test_1 )
{
    mpf_t mpf1;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 0.010203);
    MultiPrecision (300);
    MultiPrecision mp (0.010203);
    EXPECT_EQ (mpf1 , mp);
}

TEST( MultiPrecisionTest, test_2 )
{
    mpf_t mpf1;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 0.010203);
    mpf_neg(mpf1, mpf1);
    MultiPrecision (300);
    MultiPrecision mp (0.010203);
    MultiPrecision res (-mp);
    EXPECT_EQ (mpf1 , res);
}

TEST( MultiPrecisionTest, test_3 )
{
    mpf_t mpf1, mpf2;
    mpf_set_default_prec (300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init(mpf2);
    mpf_add(mpf2,mpf1,mpf1);
    MultiPrecision (300);
    MultiPrecision mp (1.123456);
    MultiPrecision res (mp+=mp);
    EXPECT_EQ (mpf2 , res);
}

TEST( MultiPrecisionTest, test_4 )
{
    mpf_t mpf1, mpf2;
    mpf_set_default_prec (300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init(mpf2);
    mpf_sub(mpf2,mpf1,mpf1);
    MultiPrecision (300);
    MultiPrecision mp (1.123456);
    MultiPrecision res (mp-=mp);
    EXPECT_EQ (mpf2 , res);
}

TEST( MultiPrecisionTest, test_5 )
{
    mpf_t mpf1, mpf2;
    mpf_set_default_prec (300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init(mpf2);
    mpf_mul(mpf2,mpf1,mpf1);
    MultiPrecision (300);
    MultiPrecision mp (1.123456);
    MultiPrecision res (mp*=mp);
    EXPECT_EQ (mpf2 , res);
}

TEST( MultiPrecisionTest, test_6 )
{
    mpf_t mpf1, mpf2;
    mpf_set_default_prec (300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init(mpf2);
    mpf_div(mpf2,mpf1,mpf1);
    MultiPrecision (300);
    MultiPrecision mp (1.123456);
    MultiPrecision res (mp/=mp);
    EXPECT_EQ (mpf2 , res);
}

TEST( MultiPrecisionTest, test_7 )
{
    mpf_t mpf1, mpf2, mpf3;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init_set_d(mpf2, 2.123456);
    mpf_init(mpf3);
    mpf_add(mpf3, mpf1, mpf2);
    MultiPrecision (300);
    MultiPrecision m1 (1.123456);
    MultiPrecision m2 (2.123456);
    MultiPrecision res (m1+m2);
    EXPECT_EQ (mpf3 , res);
}

TEST( MultiPrecisionTest, test_8 )
{
    mpf_t mpf1, mpf2, mpf3;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init_set_d(mpf2, 2.123456);
    mpf_init(mpf3);
    mpf_sub(mpf3, mpf1, mpf2);
    MultiPrecision (300);
    MultiPrecision m1 (1.123456);
    MultiPrecision m2 (2.123456);
    MultiPrecision res (m1-m2);
    EXPECT_EQ (mpf3 , res);
}

TEST( MultiPrecisionTest, test_9 )
{
    mpf_t mpf1, mpf2, mpf3;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init_set_d(mpf2, 2.123456);
    mpf_init(mpf3);
    mpf_div(mpf3, mpf1, mpf2);
    MultiPrecision (300);
    MultiPrecision m1 (1.123456);
    MultiPrecision m2 (2.123456);
    MultiPrecision res (m1/m2);
    EXPECT_EQ (mpf3 , res);
}

TEST( MultiPrecisionTest, test_10 )
{
    mpf_t mpf1, mpf2, mpf3;
    mpf_set_default_prec(300);
    mpf_init_set_d(mpf1, 1.123456);
    mpf_init_set_d(mpf2, 2.123456);
    mpf_init(mpf3);
    mpf_mul(mpf3, mpf1, mpf2);
    MultiPrecision (300);
    MultiPrecision m1 (1.123456);
    MultiPrecision m2 (2.123456);
    MultiPrecision res (m1*m2);
    EXPECT_EQ (mpf3 , res);
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