/***************************************************************************
                          ListTest.cpp  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

#include <TNL/Containers/List.h>

using namespace TNL;
using namespace TNL::Containers;


// test fixture for typed tests
template< typename List >
class ListTest : public ::testing::Test
{
protected:
   using ListType = List;
};

// types for which ListTest is instantiated
using ListTypes = ::testing::Types<
   List< short  >,
   List< int    >,
   List< long   >,
   List< float  >,
   List< double >,
   List< String >
>;

TYPED_TEST_CASE( ListTest, ListTypes );


TYPED_TEST( ListTest, constructor )
{
   using ListType = typename TestFixture::ListType;

   ListType list;
   EXPECT_TRUE( list.isEmpty() );
   EXPECT_EQ( list.getSize(), 0 );

   list.Append( 0 );
   EXPECT_EQ( list.getSize(), 1 );

   ListType copy( list );
   list.Append( 0 );
   EXPECT_EQ( list.getSize(), 2 );
   EXPECT_EQ( copy.getSize(), 1 );
   EXPECT_EQ( copy[ 0 ], list[ 0 ] );
}

TYPED_TEST( ListTest, operations )
{
   using ListType = typename TestFixture::ListType;
   using ElementType = typename ListType::ElementType;

   ListType a, b;

   a.Append( 0 );
   a.Append( 1 );
   a.Prepend( 2 );
   a.Insert( 3, 1 );
   EXPECT_EQ( a.getSize(), 4 );
   EXPECT_EQ( a[ 0 ], (ElementType) 2 );
   EXPECT_EQ( a[ 1 ], (ElementType) 3 );
   EXPECT_EQ( a[ 2 ], (ElementType) 0 );
   EXPECT_EQ( a[ 3 ], (ElementType) 1 );

   b = a;
   EXPECT_EQ( b.getSize(), 4 );
   EXPECT_EQ( a, b );

   b.Insert( 4, 4 );
   EXPECT_NE( a, b );
   EXPECT_EQ( b[ 4 ], (ElementType) 4 );

   a.AppendList( b );
   EXPECT_EQ( a.getSize(), 9 );
   EXPECT_EQ( a[ 0 ], (ElementType) 2 );
   EXPECT_EQ( a[ 1 ], (ElementType) 3 );
   EXPECT_EQ( a[ 2 ], (ElementType) 0 );
   EXPECT_EQ( a[ 3 ], (ElementType) 1 );
   EXPECT_EQ( a[ 4 ], (ElementType) 2 );
   EXPECT_EQ( a[ 5 ], (ElementType) 3 );
   EXPECT_EQ( a[ 6 ], (ElementType) 0 );
   EXPECT_EQ( a[ 7 ], (ElementType) 1 );
   EXPECT_EQ( a[ 8 ], (ElementType) 4 );

   a.PrependList( b );
   EXPECT_EQ( a.getSize(), 14 );
   EXPECT_EQ( a[ 0 ],  (ElementType) 2 );
   EXPECT_EQ( a[ 1 ],  (ElementType) 3 );
   EXPECT_EQ( a[ 2 ],  (ElementType) 0 );
   EXPECT_EQ( a[ 3 ],  (ElementType) 1 );
   EXPECT_EQ( a[ 4 ],  (ElementType) 4 );
   EXPECT_EQ( a[ 5 ],  (ElementType) 2 );
   EXPECT_EQ( a[ 6 ],  (ElementType) 3 );
   EXPECT_EQ( a[ 7 ],  (ElementType) 0 );
   EXPECT_EQ( a[ 8 ],  (ElementType) 1 );
   EXPECT_EQ( a[ 9 ],  (ElementType) 2 );
   EXPECT_EQ( a[ 10 ], (ElementType) 3 );
   EXPECT_EQ( a[ 11 ], (ElementType) 0 );
   EXPECT_EQ( a[ 12 ], (ElementType) 1 );
   EXPECT_EQ( a[ 13 ], (ElementType) 4 );
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
