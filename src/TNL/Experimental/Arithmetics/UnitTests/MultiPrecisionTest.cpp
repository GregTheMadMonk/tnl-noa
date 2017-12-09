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
TEST (MultiPrecisionTest, number_assignment)
{
    /* GMPLIB */
    mpf_t mpf1;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 0.010203);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (0.010203);
    
    EXPECT_EQ (mp1 , mpf1);
}


TEST (MultiPrecisionTest, number_negation)
{
    /* GMPLIB */
    mpf_t mpf1, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 0.010203);
    mpf_init (GMP_res);
    mpf_neg (GMP_res , mpf1);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (0.010203);
    MultiPrecision MP_res (-mp1);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_plus_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_add (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 += mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_minus_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_sub (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 -= mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_mul_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_mul (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 *= mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_div_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_div (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 /= mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_plus)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_add (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 + mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_minus)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_sub (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 - mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_div)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_div (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 / mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_mul)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (300);
    mpf_init_set_d (mpf1 , 1.123456);
    mpf_init_set_d (mpf2 , 0.010203);
    mpf_init (GMP_res);
    mpf_mul (GMP_res , mpf1 , mpf2);
    
    /* MultiPrecision */
    MultiPrecision (300);
    MultiPrecision mp1 (1.123456);
    MultiPrecision mp2 (0.010203);
    MultiPrecision MP_res (mp1 * mp2);
    
    EXPECT_EQ (MP_res , GMP_res);
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