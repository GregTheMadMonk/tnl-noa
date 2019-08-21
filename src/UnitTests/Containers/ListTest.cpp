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

// minimal custom data structure usable as ValueType in List
struct MyData
{
   double data;

   __cuda_callable__
   MyData() : data(0) {}

   template< typename T >
   __cuda_callable__
   MyData( T v ) : data(v) {}

   __cuda_callable__
   bool operator==( const MyData& v ) const { return data == v.data; }

   __cuda_callable__
   bool operator!=( const MyData& v ) const { return data != v.data; }
};

std::ostream& operator<<( std::ostream& str, const MyData& v )
{
   return str << v.data;
}


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
   List< MyData >
>;

TYPED_TEST_SUITE( ListTest, ListTypes );


TYPED_TEST( ListTest, constructor )
{
   using ListType = typename TestFixture::ListType;
   using ValueType = typename ListType::ValueType;

   ListType list;
   EXPECT_TRUE( list.isEmpty() );
   EXPECT_EQ( list.getSize(), 0 );

   list.Append( ( ValueType ) 0 );
   EXPECT_EQ( list.getSize(), 1 );

   ListType copy( list );
   list.Append( ( ValueType ) 0 );
   EXPECT_EQ( list.getSize(), 2 );
   EXPECT_EQ( copy.getSize(), 1 );
   EXPECT_EQ( copy[ 0 ], list[ 0 ] );
}

TYPED_TEST( ListTest, operations )
{
   using ListType = typename TestFixture::ListType;
   using ValueType = typename ListType::ValueType;

   ListType a, b;

   a.Append( (ValueType) 0 );
   a.Append( (ValueType) 1 );
   a.Prepend( (ValueType) 2 );
   a.Insert( (ValueType) 3, 1 );
   EXPECT_EQ( a.getSize(), 4 );
   EXPECT_EQ( a[ 0 ], (ValueType) 2 );
   EXPECT_EQ( a[ 1 ], (ValueType) 3 );
   EXPECT_EQ( a[ 2 ], (ValueType) 0 );
   EXPECT_EQ( a[ 3 ], (ValueType) 1 );

   b = a;
   EXPECT_EQ( b.getSize(), 4 );
   EXPECT_EQ( a, b );

   b.Insert( ( ValueType ) 4, 4 );
   EXPECT_NE( a, b );
   EXPECT_EQ( b[ 4 ], (ValueType) 4 );

   a.AppendList( b );
   EXPECT_EQ( a.getSize(), 9 );
   EXPECT_EQ( a[ 0 ], (ValueType) 2 );
   EXPECT_EQ( a[ 1 ], (ValueType) 3 );
   EXPECT_EQ( a[ 2 ], (ValueType) 0 );
   EXPECT_EQ( a[ 3 ], (ValueType) 1 );
   EXPECT_EQ( a[ 4 ], (ValueType) 2 );
   EXPECT_EQ( a[ 5 ], (ValueType) 3 );
   EXPECT_EQ( a[ 6 ], (ValueType) 0 );
   EXPECT_EQ( a[ 7 ], (ValueType) 1 );
   EXPECT_EQ( a[ 8 ], (ValueType) 4 );

   a.PrependList( b );
   EXPECT_EQ( a.getSize(), 14 );
   EXPECT_EQ( a[ 0 ],  (ValueType) 2 );
   EXPECT_EQ( a[ 1 ],  (ValueType) 3 );
   EXPECT_EQ( a[ 2 ],  (ValueType) 0 );
   EXPECT_EQ( a[ 3 ],  (ValueType) 1 );
   EXPECT_EQ( a[ 4 ],  (ValueType) 4 );
   EXPECT_EQ( a[ 5 ],  (ValueType) 2 );
   EXPECT_EQ( a[ 6 ],  (ValueType) 3 );
   EXPECT_EQ( a[ 7 ],  (ValueType) 0 );
   EXPECT_EQ( a[ 8 ],  (ValueType) 1 );
   EXPECT_EQ( a[ 9 ],  (ValueType) 2 );
   EXPECT_EQ( a[ 10 ], (ValueType) 3 );
   EXPECT_EQ( a[ 11 ], (ValueType) 0 );
   EXPECT_EQ( a[ 12 ], (ValueType) 1 );
   EXPECT_EQ( a[ 13 ], (ValueType) 4 );
}
#endif


#include "../main.h"
