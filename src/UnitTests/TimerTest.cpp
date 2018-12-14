/***************************************************************************
                          FileNameTest.cpp  -  description
                             -------------------
    begin                : Oct 17, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by Nina Dzugasova

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/Timer.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( TimerTest, Constructor )
{
    Timer time;
    time.reset();
    EXPECT_EQ(time.getRealTime(),0);
    /*time.start();
    EXPECT_FALSE(time.stopState);

    time.stop();
    EXPECT_TRUE(time.stopState);

    EXPECT_NE(time.getRealTime(),0);*/
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


