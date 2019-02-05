#include "gtest/gtest.h"

#include <TNL/Containers/NDArray.h>

using namespace TNL::Containers;
using std::index_sequence;

template< typename Array >
void expect_identity( const Array& a )
{
    Array identity;
    identity.setLike( a );
    int last = 0;
    for( int i = 0; i < identity.getSize(); i++ ) {
        // skip negative/invalid entries due to alignment
        if( a[ i ] < 0 )
            identity[ i ] = a[ i ];
        else
            identity[ i ] = last++;
    }
    EXPECT_EQ( a, identity );
}

TEST( NDArrayTest, setLike )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );

    decltype(a) b;
    EXPECT_EQ( b.template getSize< 0 >(), 0 );
    EXPECT_EQ( b.template getSize< 1 >(), 0 );
    EXPECT_EQ( b.template getSize< 2 >(), 0 );
    EXPECT_EQ( b.template getSize< 3 >(), 0 );
    EXPECT_EQ( b.template getSize< 4 >(), 0 );
    EXPECT_EQ( b.template getSize< 5 >(), 0 );
    b.setLike( a );
    EXPECT_EQ( b.template getSize< 0 >(), I );
    EXPECT_EQ( b.template getSize< 1 >(), J );
    EXPECT_EQ( b.template getSize< 2 >(), K );
    EXPECT_EQ( b.template getSize< 3 >(), L );
    EXPECT_EQ( b.template getSize< 4 >(), M );
    EXPECT_EQ( b.template getSize< 5 >(), N );
}

TEST( NDArrayTest, reset )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    EXPECT_EQ( a.template getSize< 0 >(), I );
    EXPECT_EQ( a.template getSize< 1 >(), J );
    EXPECT_EQ( a.template getSize< 2 >(), K );
    EXPECT_EQ( a.template getSize< 3 >(), L );
    EXPECT_EQ( a.template getSize< 4 >(), M );
    EXPECT_EQ( a.template getSize< 5 >(), N );

    a.reset();
    EXPECT_EQ( a.template getSize< 0 >(), 0 );
    EXPECT_EQ( a.template getSize< 1 >(), 0 );
    EXPECT_EQ( a.template getSize< 2 >(), 0 );
    EXPECT_EQ( a.template getSize< 3 >(), 0 );
    EXPECT_EQ( a.template getSize< 4 >(), 0 );
    EXPECT_EQ( a.template getSize< 5 >(), 0 );
}

TEST( NDArrayTest, Static_1D )
{
    constexpr int I = 3;
    NDArray< int, SizesHolder< int, I > > a;
    a.setSizes( 0 );

    int v = 0;
    for( int i = 0; i < I; i++ ) {
        a( i ) = v++;
        EXPECT_EQ( a[ i ], a( i ) );
    }

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Static_2D_Identity )
{
    constexpr int I = 3, J = 5;
    NDArray< int, SizesHolder< int, I, J > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int i = 0; i < I; i++ )
        for( int j = 0; j < J; j++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Static_2D_Permuted )
{
    constexpr int I = 3, J = 5;
    NDArray< int,
             SizesHolder< int, I, J >,
             index_sequence< 1, 0 > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int j = 0; j < J; j++ )
        for( int i = 0; i < I; i++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Dynamic_6D )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );

    // initialize entries invalid due to alignment to -1
    a.getStorageArray().setValue( -1 );

    int v = 0;
    for( int n = 0; n < N; n++ )
        for( int l = 0; l < L; l++ )
            for( int m = 0; m < M; m++ )
                for( int k = 0; k < K; k++ )
                    for( int i = 0; i < I; i++ )
                        for( int j = 0; j < J; j++ )
                            a( i, j, k, l, m, n ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, CopySemantics )
{
    constexpr int I = 3, J = 4;
    NDArray< int, SizesHolder< int, I, J > > a, b, c;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        a( i, j ) = v++;

    expect_identity( a.getStorageArray() );

    b = a;
    EXPECT_EQ( a, b );

    auto a_view = a.getView();
    auto b_view = b.getView();
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a_view.getView(), b_view );
    EXPECT_EQ( a_view.getConstView(), b_view.getConstView() );
    EXPECT_EQ( a.getConstView(), b.getConstView() );
    EXPECT_EQ( a.getConstView(), b_view.getConstView() );

    c.setSizes( 0, 0 );
    auto c_view = c.getView();
    c_view = b_view;
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );
}

TEST( NDArrayTest, SizesHolderPrinter )
{
   SizesHolder< int, 0, 1, 2 > holder;
   holder.setSize< 0 >( 3 );

   std::stringstream str;
   str << holder;
   EXPECT_EQ( str.str(), "SizesHolder< 0, 1, 2 >( 3, 1, 2 )" );
}

TEST( NDArrayTest, forAll_dynamic )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) = 1;
    };

    a.forAll( setter );

    for( int n = 0; n < N; n++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j, k, l, m, n ), 1 );
}

TEST( NDArrayTest, forAll_static )
{
    constexpr int I = 3, J = 4;
    NDArray< int, SizesHolder< int, I, J > > a;
    a.setSizes( 0, 0 );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        a( i, j ) = 0;

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) = 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j ), 1 );
}

//#include "GtestMissingError.h"
int main( int argc, char* argv[] )
{
//#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
//#else
//   throw GtestMissingError();
//#endif
}
