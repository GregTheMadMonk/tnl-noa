/***************************************************************************
                          StaticVectorTest.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_GTEST
#include <TNL/Containers/StaticVector.h>

#include "gtest/gtest.h"

using namespace TNL;

const int Size( 16 );
typedef double RealType;

TEST( StaticVectorTest, testOperators )
{
   Containers::StaticVector< Size, RealType > u1( 1.0 ), u2( 2.0 ), u3( 3.0 );

   u1 += u2;
   ASSERT_TRUE( u1[ 0 ] == 3.0 );
   ASSERT_TRUE( u1[ Size - 1 ] == 3.0 );

   u1 -= u2;
   ASSERT_TRUE( u1[ 0 ] == 1.0 );
   ASSERT_TRUE( u1[ Size - 1 ] == 1.0 );

   u1 *= 2.0;
   ASSERT_TRUE( u1[ 0 ] == 2.0 );
   ASSERT_TRUE( u1[ Size - 1 ] == 2.0 );

   u3 = u1 + u2;
   ASSERT_TRUE( u3[ 0 ] == 4.0 );
   ASSERT_TRUE( u3[ Size - 1 ] == 4.0 );

   u3 = u1 - u2;
   ASSERT_TRUE( u3[ 0 ] == 0.0 );
   ASSERT_TRUE( u3[ Size - 1 ] == 0.0 );

   u3 = u1 * 2.0;
   ASSERT_TRUE( u3[ 0 ] == 4.0 );
   ASSERT_TRUE( u3[ Size - 1 ] == 4.0 );

   ASSERT_TRUE( u1 * u2 == 4.0 * Size );

   ASSERT_TRUE( u1 < u3 );
   ASSERT_TRUE( u1 <= u3 );
   ASSERT_TRUE( u1 <= u2 );
   ASSERT_TRUE( u3 > u1 );
   ASSERT_TRUE( u3 >= u1 );
   ASSERT_TRUE( u2 >= u1 );
}
#endif


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
