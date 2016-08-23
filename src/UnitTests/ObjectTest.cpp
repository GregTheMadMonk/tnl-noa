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

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( ObjectTest, SaveAndLoadTest )
{
   Object testObject;
   File file;
   file.open( "test-file.tnl", tnlWriteMode );
   ASSERT_TRUE( testObject.save( file ) );
   file.close();
   file.open( "test-file.tnl", tnlReadMode );
   ASSERT_TRUE( testObject.load( file ) );
}
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


