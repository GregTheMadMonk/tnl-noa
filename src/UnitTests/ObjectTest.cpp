/***************************************************************************
                          ObjectTest.cpp  -  description
                             -------------------
    begin                : Jul 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Host.h>
#include <TNL/Object.h>
#include <TNL/File.h>
#include <TNL/Containers/Array.h>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( ObjectTest, SaveAndLoadTest )
{
   Object testObject;
   File file;
   file.open( "test-file.tnl", IOMode::write );
   ASSERT_TRUE( testObject.save( file ) );
   file.close();
   file.open( "test-file.tnl", IOMode::read );
   ASSERT_TRUE( testObject.load( file ) );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

TEST( ObjectTest, parseObjectTypeTest )
{
   Containers::List< String > parsed;
   Containers::List< String > expected;

   // plain type
   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "int", parsed ) );
   expected.Append( "int" );
   EXPECT_EQ( parsed, expected );

   // type with space
   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "short int", parsed ) );
   expected.Append( "short int" );
   EXPECT_EQ( parsed, expected );

   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "unsigned short int", parsed ) );
   expected.Append( "unsigned short int" );
   EXPECT_EQ( parsed, expected );

   // composed type
   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "Containers::Vector< double, Devices::Host, int >", parsed ) );
   expected.Append( "Containers::Vector" );
   expected.Append( "double" );
   expected.Append( "Devices::Host" );
   expected.Append( "int" );
   EXPECT_EQ( parsed, expected );

   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "Containers::Vector< Containers::List< String >, Devices::Host, int >", parsed ) );
   expected.Append( "Containers::Vector" );
   expected.Append( "Containers::List< String >" );
   expected.Append( "Devices::Host" );
   expected.Append( "int" );
   EXPECT_EQ( parsed, expected );

   // spaces in the template parameter
   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "A< short int >", parsed ) );
   expected.Append( "A" );
   expected.Append( "short int" );
   EXPECT_EQ( parsed, expected );

   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "A< B< short int >, C >", parsed ) );
   expected.Append( "A" );
   expected.Append( "B< short int >" );
   expected.Append( "C" );
   EXPECT_EQ( parsed, expected );

   // spaces at different places in the template parameter
   parsed.reset();
   expected.reset();
   ASSERT_TRUE( parseObjectType( "A< b , c <E>  ,d>", parsed ) );
   expected.Append( "A" );
   expected.Append( "b" );
   expected.Append( "c <E>" );
   expected.Append( "d" );
   EXPECT_EQ( parsed, expected );
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
