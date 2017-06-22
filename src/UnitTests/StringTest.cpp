/***************************************************************************
                          StringTest.cpp  -  description
                             -------------------
    begin                : Jul 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

#include <TNL/String.h>
#include <TNL/File.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( StringTest, BasicConstructor )
{
   String str;
   ASSERT_EQ( strcmp( str. getString(), "" ), 0 );
}

TEST( StringTest, ConstructorWithChar )
{
   String str1( "string1" );
   String str2( "xxxstring2", 3 );
   String str3( "string3xxx", 0, 3 );
   String str4( "xxxstring4xxx", 3, 3 );

   ASSERT_EQ( strcmp( str1. getString(), "string1" ), 0 );
   ASSERT_EQ( strcmp( str2. getString(), "string2" ), 0 );
   ASSERT_EQ( strcmp( str3. getString(), "string3" ), 0 );
   ASSERT_EQ( strcmp( str4. getString(), "string4" ), 0 );
}

TEST( StringTest, CopyConstructor )
{
   String string( "string1" );
   String emptyString( "" );
   String string2( string );
   String emptyString2( emptyString );

   ASSERT_EQ( strcmp( string2. getString(), "string1" ), 0 );
   ASSERT_EQ( strcmp( emptyString2. getString(), "" ), 0 );
}

TEST( StringTest, ConstructorWithNumber )
{
   String string1( 10 );
   String string2( -5 );

   ASSERT_EQ( strcmp( string1. getString(), "10" ), 0 );
   ASSERT_EQ( strcmp( string2. getString(), "-5" ), 0 );
}

TEST( StringTest, SetString )
{
   String str1, str2, str3, str4;

   str1. setString( "string1" );
   str2. setString( "xxxstring2", 3 );
   str3. setString( "string3xxx", 0, 3 );
   str4. setString( "xxxstring4xxx", 3, 3 );

   ASSERT_EQ( strcmp( str1. getString(), "string1" ), 0 );
   ASSERT_EQ( strcmp( str2. getString(), "string2" ), 0 );
   ASSERT_EQ( strcmp( str3. getString(), "string3" ), 0 );
   ASSERT_EQ( strcmp( str4. getString(), "string4" ), 0 );
}

TEST( StringTest, IndexingOperator )
{
   String str( "1234567890" );
   ASSERT_EQ( str[ 0 ], '1' );
   ASSERT_EQ( str[ 1 ], '2' );
   ASSERT_EQ( str[ 2 ], '3' );
   ASSERT_EQ( str[ 3 ], '4' );
   ASSERT_EQ( str[ 4 ], '5' );
   ASSERT_EQ( str[ 5 ], '6' );
   ASSERT_EQ( str[ 6 ], '7' );
   ASSERT_EQ( str[ 7 ], '8' );
   ASSERT_EQ( str[ 8 ], '9' );
   ASSERT_EQ( str[ 9 ], '0' );
}

TEST( StringTest, AssignmentOperator )
{
   String string1( "string" );
   String string2;
   string2 = string1;

   ASSERT_EQ( strcmp( string2. getString(), "string" ), 0 );
}

TEST( StringTest, AdditionAssignmentOperator )
{
   String string1( "string" );
   String string2;
   string2 = string1;
   string2 += "string2";

   ASSERT_EQ( strcmp( string2. getString(), "stringstring2" ), 0 );
}

TEST( StringTest, strip )
{
   EXPECT_EQ( String( "string" ).strip(), String( "string" ) );
   EXPECT_EQ( String( "  string" ).strip(), String( "string" ) );
   EXPECT_EQ( String( "string  " ).strip(), String( "string" ) );
   EXPECT_EQ( String( "  string  " ).strip(), String( "string" ) );
   EXPECT_EQ( String( " string1  string2  " ).strip(), String( "string1  string2" ) );
   EXPECT_EQ( String( "" ).strip(), String( "" ) );
   EXPECT_EQ( String( "  " ).strip(), String( "" ) );
}


TEST( StringTest, SaveLoad )
{
   String str1( "testing-string" );
   File file;
   file.open( "test-file.tnl", IOMode::write );
   ASSERT_TRUE( str1.save( file ) );
   file.close();
   file.open( "test-file.tnl", IOMode::read );
   String str2;
   ASSERT_TRUE( str2.load( file ) );
   ASSERT_EQ( str1, str2 );
};

#endif


int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}

