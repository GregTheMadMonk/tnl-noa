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

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( FileNameTest, Constructor )
{
   /*String str1( "string1" );
   String str2( "xxxstring2", 3 );
   String str3( "string3xxx", 0, 3 );
   String str4( "xxxstring4xxx", 3, 3 );

   EXPECT_EQ( strcmp( str1.getString(), "string1" ), 0 );
   EXPECT_EQ( strcmp( str2.getString(), "string2" ), 0 );
   EXPECT_EQ( strcmp( str3.getString(), "string3" ), 0 );
   EXPECT_EQ( strcmp( str4.getString(), "string4" ), 0 );*/
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

