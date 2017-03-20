/***************************************************************************
                          StaticArrayTester.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Array.h>

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

using namespace TNL;
using namespace TNL::Containers;

class testingClassForStaticArrayTester
{
   public:

      static String getType()
      {
         return String( "testingClassForStaticArrayTester" );
      };
};

String getType( const testingClassForStaticArrayTester& c )
{
   return String( "testingClassForStaticArrayTester" );
};

#ifdef HAVE_GTEST

typedef int ElementType;
const int Size( 16 );


TEST( StaticArrayTest, testConstructors )
{
   ElementType data[ Size ];
   for( int i = 0; i < Size; i++ )
      data[ i ] = i;

   StaticArray< Size, ElementType > u1( data );
   for( int i = 0; i < Size; i++ )
      ASSERT_EQ( u1[ i ], data[ i ] );

   StaticArray< Size, ElementType > u2( 7 );
   for( int i = 0; i < Size; i++ )
      ASSERT_EQ( u2[ i ], 7 );

   StaticArray< Size, ElementType > u3( u1 );
   for( int i = 0; i < Size; i++ )
      ASSERT_EQ( u3[ i ], u1[ i ] );
}

template< typename Element >
void checkCoordinates( const StaticArray< 1, Element >& u )
{
   ASSERT_EQ( u.x(), 0 );
}

template< typename Element >
void checkCoordinates( const StaticArray< 2, Element >& u )
{
   ASSERT_EQ( u.x(), 0 );
   ASSERT_EQ( u.y(), 1 );
}

template< typename Element >
void checkCoordinates( const StaticArray< 3, Element >& u )
{
   ASSERT_EQ( u.x(), 0 );
   ASSERT_EQ( u.y(), 1 );
   ASSERT_EQ( u.z(), 2 );
}

template< int _Size, typename Element >
void checkCoordinates( const StaticArray< _Size, Element >& u )
{
}

TEST( StaticArrayTest, testCoordinatesGetter )
{
   StaticArray< Size, ElementType > u;
   for( int i = 0; i < Size; i++ )
      u[ i ] = i;

   checkCoordinates( u );
}

TEST( StaticArrayTest, testComparisonOperator )
{
   StaticArray< Size, ElementType > u1, u2, u3;

   for( int i = 0; i < Size; i++ )
   {
      u1[ i ] = 1;
      u2[ i ] = i;
      u3[ i ] = i;
   }

   ASSERT_TRUE( u1 == u1 );
   ASSERT_TRUE( u1 != u2 );
   ASSERT_TRUE( u2 == u3 );
}

TEST( StaticArrayTest, testAssignmentOperator )
{
   StaticArray< Size, ElementType > u1, u2, u3;

   for( int i = 0; i < Size; i++ )
   {
      u1[ i ] = 1;
      u2[ i ] = i;
   }

   u3 = u1;
   ASSERT_TRUE( u3 == u1 );
   ASSERT_TRUE( u3 != u2 );

   u3 = u2;
   ASSERT_TRUE( u3 == u2 );
   ASSERT_TRUE( u3 != u1 );
}

TEST( StaticArrayTest, testLoadAndSave )
{
   StaticArray< Size, ElementType > u1( 7 ), u2( 0 );
   File file;
   file.open( "tnl-static-array-test.tnl", tnlWriteMode );
   u1.save( file );
   file.close();
   file.open( "tnl-static-array-test.tnl", tnlReadMode );
   u2.load( file );
   file.close();

   ASSERT_EQ( u1, u2 );
}

TEST( StaticArrayTest, testSort )
{
   StaticArray< Size, ElementType > u;
   for( int i = 0; i < Size; i++ )
      u[ i ] = Size - i - 1;
   u.sort();

   for( int i = 0; i < Size; i++ )
      ASSERT_EQ( u[ i ], i );
}

TEST( StaticArrayTest, testStreamOperator )
{
   StaticArray< Size, ElementType > u;
   std::stringstream testStream;
   testStream << u;
}

TEST( StaticArrayTest, testBindToArray )
{
   StaticArray< Size, ElementType > a;
   for( int i = 0; i < Size; i++ )
      a[ i ] = i+1;

   Array< ElementType, Devices::Host > sharedArray;
   sharedArray.bind( a );
   for( int i = 0; i < Size; i++ )
      ASSERT_EQ( a[ i ], sharedArray[ i ] );

}
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

