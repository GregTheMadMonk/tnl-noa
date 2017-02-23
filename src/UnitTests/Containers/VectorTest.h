/***************************************************************************
                          VectorTester.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/File.h>
#include <TNL/Math.h>

using namespace TNL;

#ifdef HAVE_GTEST

TEST( VectorTest, testMax )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v. setElement( i, i );
   ASSERT_TRUE( v. max() == 9 );
};

TEST( VectorTest, testMin )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v. setElement( i, i );
   ASSERT_TRUE( v. min() == 0 );
};

TEST( VectorTest, testAbsMax )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v.setElement( i, -i );
   ASSERT_TRUE( v. absMax() == 9 );
};

TEST( VectorTest, testAbsMin )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v.setElement( i,  -i );
   ASSERT_TRUE( v. absMin() == 0 );
};

TEST( VectorTest, testLpNorm )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v.setElement(  i, -2 );
   ASSERT_TRUE( isSmall( v.lpNorm( 1 ) - 20.0 ) );
   ASSERT_TRUE( isSmall( v.lpNorm( 2 ) - ::sqrt( 40.0 ) ) );
   ASSERT_TRUE( isSmall( v.lpNorm( 3 ) - ::pow( 80.0, 1.0/3.0 ) ) );
};

TEST( VectorTest, testSum )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v.setElement( i, -2 );
   ASSERT_TRUE( v. sum() == -20.0 );
   for( int i = 0; i < 10; i ++ )
      v.setElement( i,  2 );
   ASSERT_TRUE( v. sum() == 20.0 );

};

TEST( VectorTest, testDifferenceMax )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i,  i );
      v2.setElement( i, -i );
   }
   ASSERT_TRUE( v1. differenceMax( v2 ) == 18.0 );
};

TEST( VectorTest, testDifferenceMin )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, i );
      v2.setElement( i, -i );
   }
   ASSERT_TRUE( v1. differenceMin( v2 ) == 0.0 );
};

TEST( VectorTest, testDifferenceAbsMax )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, -i );
      v2.setElement( i, i );
   }
   ASSERT_TRUE( v1. differenceAbsMax( v2 ) == 18.0 );
};

TEST( VectorTest, testDifferenceAbsMin )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, -i );
      v2.setElement( i, i );
   }
   ASSERT_TRUE( v1. differenceAbsMin( v2 ) == 0.0 );
};

TEST( VectorTest, testDifferenceLpNorm )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, -1 );
      v2.setElement( i, 1 );
   }
   ASSERT_TRUE( isSmall( v1.differenceLpNorm( v2, 1.0 ) - 20.0 ) );
   ASSERT_TRUE( isSmall( v1.differenceLpNorm( v2, 2.0 ) - ::sqrt( 40.0 ) ) );
   ASSERT_TRUE( isSmall( v1.differenceLpNorm( v2, 3.0 ) - ::pow( 80.0, 1.0/3.0 ) ) );
};

TEST( VectorTest, testDifferenceSum )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, -1 );
      v2.setElement( i, 1 );
   }
   ASSERT_TRUE( v1. differenceSum( v2 ) == -20.0 );
};

TEST( VectorTest, testScalarMultiplication )
{
   Containers::Vector< RealType, Device, IndexType > v;
   v. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
      v.setElement( i, i );
   v. scalarMultiplication( 5.0 );

   for( int i = 0; i < 10; i ++ )
      ASSERT_TRUE( v. getElement( i ) == 5 * i );
};

TEST( VectorTest, testScalarProduct )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   v1.setElement( 0, -1 );
   v2.setElement( 0, 1 );
   for( int i = 1; i < 10; i ++ )
   {
      v1.setElement( i, v1.getElement( i - 1 ) * -1 );
      v2.setElement( i, v2.getElement( i - 1 ) );
   }
   ASSERT_TRUE( v1. scalarProduct( v2 ) == 0.0 );
};

TEST( VectorTest, addVectorTest )
{
   Containers::Vector< RealType, Device, IndexType > v1, v2;
   v1. setSize( 10 );
   v2. setSize( 10 );
   for( int i = 0; i < 10; i ++ )
   {
      v1.setElement( i, i );
      v2.setElement( i, 2.0 * i );
   }
   v1. addVector( v2, 2.0 );
   for( int i = 0; i < 10; i ++ )
      ASSERT_TRUE( v1. getElement( i ) == 5.0 * i );
};

#endif /* HAVE_GTEST */

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}
