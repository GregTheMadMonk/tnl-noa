/**************************************************
* filename:		QuadTest.cpp	          *
* created:		October 27, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/Experimental/Arithmetics/MultiPrecision.h>
#include <TNL/Experimental/Arithmetics/Quad.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( QuadTest, test_1 )
{
    Quad<double> qd (0.010203040506);
    MultiPrecision mp (0.010203040506);
    EXPECT_EQ (mp , qd);
}

TEST( QuadTest, test_2 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (qd1+=qd1);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (mp1+=mp1);
    EXPECT_EQ (mp2 , qd2);
}

TEST( QuadTest, test_3 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (qd1-=qd1);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (mp1-=mp1);
    EXPECT_EQ (mp2 , qd2);
}

TEST( QuadTest, test_4 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (qd1*=qd1);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (mp1*=mp1);
    EXPECT_EQ (mp2 , qd2);
}

TEST( QuadTest, test_5 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (qd1/=qd1);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (mp1/=mp1);
    EXPECT_EQ (mp2 , qd2);
}

TEST( QuadTest, test_6 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    Quad<double> QDres (qd1+qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    MultiPrecision MPres (mp1+mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_7 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    Quad<double> QDres (qd1-qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    MultiPrecision MPres (mp1-mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_8 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    Quad<double> QDres (qd1*qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    MultiPrecision MPres (mp1*mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_9 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    Quad<double> QDres (qd2*qd1);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    MultiPrecision MPres (mp2*mp1);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_10 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1==qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1==mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_11 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1!=qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1!=mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_12 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1<qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1<mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_13 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1>qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1>mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_14 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1>=qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1>=mp2);
    EXPECT_EQ (MPres , QDres);
}

TEST( QuadTest, test_15 )
{
    Quad<double> qd1 (0.010203040506);
    Quad<double> qd2 (4.102030405060);
    bool QDres (qd1<=qd2);
    MultiPrecision mp1 (0.010203040506);
    MultiPrecision mp2 (4.102030405060);
    bool MPres (mp1<=mp2);
    EXPECT_EQ (MPres , QDres);
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