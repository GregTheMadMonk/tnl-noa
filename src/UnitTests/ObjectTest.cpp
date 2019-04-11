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
   file.open( "test-file.tnl", File::Mode::Out );
   testObject.save( file );
   file.close();
   file.open( "test-file.tnl", File::Mode::In );
   testObject.load( file );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}

TEST( ObjectTest, parseObjectTypeTest )
{
   std::vector< String > parsed;
   std::vector< String > expected;

   // plain type
   parsed = parseObjectType( "int" );
   expected = {"int"};
   EXPECT_EQ( parsed, expected );

   // type with space
   parsed = parseObjectType( "short int" );
   expected = {"short int"};
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "unsigned short int" );
   expected = {"unsigned short int"};
   EXPECT_EQ( parsed, expected );

   // composed type
   parsed = parseObjectType( "Containers::Vector< double, Devices::Host, int >" );
   expected = { "Containers::Vector", "double", "Devices::Host", "int" };
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "Containers::Vector< Containers::List< String >, Devices::Host, int >" );
   expected = { "Containers::Vector", "Containers::List< String >", "Devices::Host", "int" };
   EXPECT_EQ( parsed, expected );

   // spaces in the template parameter
   parsed = parseObjectType( "A< short int >" );
   expected = { "A", "short int" };
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "A< B< short int >, C >" );
   expected = { "A", "B< short int >", "C" };
   EXPECT_EQ( parsed, expected );

   // spaces at different places in the template parameter
   parsed = parseObjectType( "A< b , c <E>  ,d>" );
   expected = { "A", "b", "c <E>", "d" };
   EXPECT_EQ( parsed, expected );
}

TEST( HeaderTest, SaveAndLoadTest )
{
   Object testObject;
   File file;
   file.open( "test-file.tnl", File::Mode::Out );
   saveHeader( file, "TYPE" );
   file.close();
   file.open( "test-file.tnl", File::Mode::In );
   String type;
   loadHeader( file, type );
   
   EXPECT_EQ( type, "TYPE" );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
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
