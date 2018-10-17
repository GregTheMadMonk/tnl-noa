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
#include <TNL/Containers/List.h>

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( StringTest, BasicConstructor )
{
   String str;
   EXPECT_EQ( strcmp( str.getString(), "" ), 0 );
}

TEST( StringTest, ConstructorWithChar )
{
   String str1( "string1" );
   String str2( "xxxstring2", 3 );
   String str3( "string3xxx", 0, 3 );
   String str4( "xxxstring4xxx", 3, 3 );

   EXPECT_EQ( strcmp( str1.getString(), "string1" ), 0 );
   EXPECT_EQ( strcmp( str2.getString(), "string2" ), 0 );
   EXPECT_EQ( strcmp( str3.getString(), "string3" ), 0 );
   EXPECT_EQ( strcmp( str4.getString(), "string4" ), 0 );
}

TEST( StringTest, CopyConstructor )
{
   String string( "string1" );
   String emptyString( "" );
   String string2( string );
   String emptyString2( emptyString );

   EXPECT_EQ( strcmp( string2.getString(), "string1" ), 0 );
   EXPECT_EQ( strcmp( emptyString2.getString(), "" ), 0 );
}

TEST( StringTest, ConstructorWithNumber )
{
   String string1( 10 );
   String string2( -5 );
   String string3( true );
   String string4( false );

   EXPECT_EQ( strcmp( string1.getString(), "10" ), 0 );
   EXPECT_EQ( strcmp( string2.getString(), "-5" ), 0 );
   EXPECT_EQ( strcmp( string3.getString(), "true" ), 0 );
   EXPECT_EQ( strcmp( string4.getString(), "false" ), 0 );
}

TEST( StringTest, GetSize )
{
    String str1( "string" );
    String str2( "12345" );
    String str3( "string3" );
    String str4( "String_4" );
    String str5( "Last String" );

    EXPECT_EQ( str1.getSize(), 6 );
    EXPECT_EQ( str2.getSize(), 5 );
    EXPECT_EQ( str3.getSize(), 7 );
    EXPECT_EQ( str4.getSize(), 8 );
    EXPECT_EQ( str5.getSize(), 11 );

    EXPECT_EQ( str1.getLength(), 6 );
    EXPECT_EQ( str2.getLength(), 5 );
    EXPECT_EQ( str3.getLength(), 7 );
    EXPECT_EQ( str4.getLength(), 8 );
    EXPECT_EQ( str5.getLength(), 11 );
}

TEST( StringTest, GetAllocatedSize )
{
    String str( "MeineKleine" );

    EXPECT_EQ( str.getAllocatedSize(), 256 );
}

TEST( StringTest, SetSize )
{
   String str;
   str.setSize( 42 );
   EXPECT_EQ( str.getAllocatedSize(), 256 );
   // it allocates one more byte for the terminating 0
   str.setSize( 256 );
   EXPECT_EQ( str.getAllocatedSize(), 512 );
}

TEST( StringTest, SetString )
{
   String str1, str2, str3, str4;

   str1.setString( "string1" );
   str2.setString( "xxxstring2", 3 );
   str3.setString( "string3xxx", 0, 3 );
   str4.setString( "xxxstring4xxx", 3, 3 );

   EXPECT_EQ( strcmp( str1.getString(), "string1" ), 0 );
   EXPECT_EQ( strcmp( str2.getString(), "string2" ), 0 );
   EXPECT_EQ( strcmp( str3.getString(), "string3" ), 0 );
   EXPECT_EQ( strcmp( str4.getString(), "string4" ), 0 );

   str4.setString( "string4_2", 0, 2 );
   EXPECT_EQ( strcmp( str4.getString(), "string4" ), 0 );
}

TEST( StringTest, GetString )
{
    String str( "MyString" );
    EXPECT_EQ( strcmp( str.getString(), "MyString" ), 0 );
}

TEST( StringTest, IndexingOperator )
{
   String str( "1234567890" );
   EXPECT_EQ( str[ 0 ], '1' );
   EXPECT_EQ( str[ 1 ], '2' );
   EXPECT_EQ( str[ 2 ], '3' );
   EXPECT_EQ( str[ 3 ], '4' );
   EXPECT_EQ( str[ 4 ], '5' );
   EXPECT_EQ( str[ 5 ], '6' );
   EXPECT_EQ( str[ 6 ], '7' );
   EXPECT_EQ( str[ 7 ], '8' );
   EXPECT_EQ( str[ 8 ], '9' );
   EXPECT_EQ( str[ 9 ], '0' );
}

TEST( StringTest, CStringOperators )
{
   // assignment operator
   String string1;
   string1 = "string";
   EXPECT_EQ( strcmp( string1.getString(), "string" ), 0 );

   // addition
   string1 += "string2";
   EXPECT_EQ( strcmp( string1.getString(), "stringstring2" ), 0 );

   // addition that forces a new page allocation
   string1 += " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long";
   EXPECT_EQ( strcmp( string1.getString(),
              "stringstring2"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long" ),
            0 );

   // addition
   EXPECT_EQ( strcmp( (String( "foo " ) + "bar").getString(), "foo bar" ), 0 );
   EXPECT_EQ( strcmp( ("foo" + String( " bar" )).getString(), "foo bar" ), 0 );

   // comparison
   EXPECT_EQ( String( "foo" ), "foo" );
   EXPECT_NE( String( "bar" ), "foo" );
   EXPECT_NE( String( "fooo" ), "foo" );
}

TEST( StringTest, StringOperators )
{
   // assignment
   String string1( "string" );
   String string2;
   string2 = string1;
   EXPECT_EQ( strcmp( string2.getString(), "string" ), 0 );

   // addition
   string1.setString( "foo " );
   string1 += String( "bar" );
   EXPECT_EQ( strcmp( string1.getString(), "foo bar" ), 0 );

   // comparison
   EXPECT_EQ( String( "foo bar" ), string1 );
   EXPECT_NE( String( "bar" ), string1 );
   EXPECT_NE( String( "bar" ), String( "baz" ) );
   EXPECT_NE( String( "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long" ),
              String( "short" ) );
   String string3( "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long" );
   string3[ 255 ] = 0;
   EXPECT_EQ( string3,
              String( "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long " ) );

   // addition
   EXPECT_EQ( String( "foo " ) + String( "bar" ), "foo bar" );
}

TEST( StringTest, SingleCharacterOperators )
{
   // assignment
   String string1;
   string1 = 'A';
   EXPECT_EQ( strcmp( string1.getString(), "A" ), 0 );

   // addition of a single character
   String string2( "string " );
   string2 += 'A';
   EXPECT_EQ( strcmp( string2.getString(), "string A" ), 0 );

   // addition of a single character that causes new page allocation
   string2.setString( "long long long long long long long long long long long long long "
                      "long long long long long long long long long long long long long "
                      "long long long long long long long long long long long long long "
                      "long long long long long long long long long long long long " );
   ASSERT_EQ( string2.getLength(), 255 );
   string2 += 'B';
   EXPECT_EQ( strcmp( string2.getString(),
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long B" ),
              0 );

   // addition
   EXPECT_EQ( strcmp( (String( "A " ) + 'B').getString(), "A B" ), 0 );
   EXPECT_EQ( strcmp( ('A' + String( " B" )).getString(), "A B" ), 0 );

   // comparison
   EXPECT_EQ( String( "A" ), 'A' );
   EXPECT_NE( String( "B" ), 'A' );
   EXPECT_NE( String( "AB" ), 'A' );
}

TEST( StringTest, CastToBoolOperator )
{
   String string;
   EXPECT_TRUE( ! string );
   EXPECT_FALSE( string );
   string.setString( "foo" );
   EXPECT_TRUE( string );
   EXPECT_FALSE( ! string );
}

TEST( StringTest, replace )
{
   EXPECT_EQ( String( "string" ).replace( "ing", "bc" ), "strbc" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT" ), "CATracadCATra" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT", 1 ), "CATracadabra" );
   EXPECT_NE( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "abracadCATra" );
   EXPECT_NE( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "abracadabra" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "CATracadCATra" );
}

TEST( StringTest, strip )
{
   EXPECT_EQ( String( "string" ).strip(), "string" );
   EXPECT_EQ( String( "  string" ).strip(), "string" );
   EXPECT_EQ( String( "string  " ).strip(), "string" );
   EXPECT_EQ( String( "  string  " ).strip(), "string" );
   EXPECT_EQ( String( " string1  string2  " ).strip(), "string1  string2" );
   EXPECT_EQ( String( "" ).strip(), "" );
   EXPECT_EQ( String( "  " ).strip(), "" );
}

TEST( StringTest, split )
{
   Containers::List< String > list;

   String( "A B C" ).split( list, ' ' );
   ASSERT_EQ( list.getSize(), 3 );
   EXPECT_EQ( list[ 0 ], "A" );
   EXPECT_EQ( list[ 1 ], "B" );
   EXPECT_EQ( list[ 2 ], "C" );

   String( "abracadabra" ).split( list, 'a' );
   ASSERT_EQ( list.getSize(), 4 );
   EXPECT_EQ( list[ 0 ], "br" );
   EXPECT_EQ( list[ 1 ], "c" );
   EXPECT_EQ( list[ 2 ], "d" );
   EXPECT_EQ( list[ 3 ], "br" );

   String( "abracadabra" ).split( list, 'b' );
   ASSERT_EQ( list.getSize(), 3 );
   EXPECT_EQ( list[ 0 ], "a" );
   EXPECT_EQ( list[ 1 ], "racada" );
   EXPECT_EQ( list[ 2 ], "ra" );

   String( "abracadabra" ).split( list, 'A' );
   ASSERT_EQ( list.getSize(), 1 );
   EXPECT_EQ( list[ 0 ], "abracadabra" );
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
   EXPECT_EQ( str1, str2 );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
};

TEST( StringTest, getLine )
{
   std::stringstream str;
   str << "Line 1" << std::endl;
   str << "Line 2" << std::endl;
   str.seekg( 0 );

   String s;

   s.getLine( str );
   EXPECT_EQ( s, "Line 1" );

   s.getLine( str );
   EXPECT_EQ( s, "Line 2" );
};

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
