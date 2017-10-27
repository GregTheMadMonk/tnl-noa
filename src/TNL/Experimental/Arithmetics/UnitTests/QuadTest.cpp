/**************************************************
* filename:		QuadTest.cpp	          *
* created:		October 27, 2017	  *
* author:		Daniel Simon	 	  *
* mail:			dansimon93@gmail.com      *
***************************************************/

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

#include <Arithmetics/QuadDouble.h>

#ifdef HAVE_GTEST 
// TODO - GTests for QuadDouble implementation
#endif

/*Main function runs the tests*/
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}