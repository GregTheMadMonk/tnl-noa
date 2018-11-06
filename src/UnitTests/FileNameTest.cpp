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

#include <TNL/FileName.h>
#include <TNL/String.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( FileNameTest, Constructor )
{
    FileName fname;

    EXPECT_EQ( fname.getFileName(), "00000." );
}

TEST( FileNameTest, Base )
{
    FileName fname;
    fname.setFileNameBase("name");

    EXPECT_EQ( fname.getFileName(), "name00000." );
}

/*TEST( FileNameTest, Extension )
{
    FileName fname;
    fname.setExtension("tnl");

    EXPECT_EQ( strcmp( fname.getFileName(), "00000.tnl" ), 0 );
}*/

/*TEST( FileNameTest, Index )
{
    FileName fname1;
    FileName fname2;
    fname1.setIndex(1);
    fname2.setIndex(50);

    EXPECT_EQ( strcmp( fname1.getFileName(), "00001." ), 0 );
    EXPECT_EQ( strcmp( fname2.getFileName(), "00050." ), 0 );
}*/

/*TEST( FileNameTest, DigitsCount )
{
    FileName fname;
    fname.setDigitsCount(4);

    EXPECT_EQ( strcmp( fname.getFileName(), "0000." ), 0 );
}

TEST( FileNameTest, AllTogether )
{
    FileName fname;
    fname.setFileNameBase("name");
    fname.setExtension("tnl");
    fname.setIndex(8);
    fname.setDigitsCount(3);

    EXPECT_EQ( strcmp( fname.getFileName(), "name008.tnl" ), 0 );
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

