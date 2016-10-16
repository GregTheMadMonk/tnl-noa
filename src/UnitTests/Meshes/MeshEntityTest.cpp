#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

#include "MeshEntityTest.h"

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}
