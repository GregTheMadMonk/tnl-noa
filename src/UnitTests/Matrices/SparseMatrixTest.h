/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>

#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>

using CSR_host_float = TNL::Matrices::CSR< float, TNL::Devices::Host, int >;
using CSR_host_int = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;

using CSR_cuda_float = TNL::Matrices::CSR< float, TNL::Devices::Cuda, int >;
using CSR_cuda_int = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>


template< typename MatrixHostFloat, typename MatrixHostInt >
void host_test_GetType()
{
    MatrixHostFloat mtrxHostFloat;
    MatrixHostInt mtrxHostInt;
    
    EXPECT_EQ( mtrxHostFloat.getType(), TNL::String( "Matrices::CSR< float, Devices::Host >" ) );
    EXPECT_EQ( mtrxHostInt.getType(), TNL::String( "Matrices::CSR< int, Devices::Host >" ) );
}

// QUESITON: Cant these two functions be combined into one? Because if no CUDA is present and we were to call
//           CUDA into the function in the TEST, to be tested, then we could have a problem.

template< typename MatrixCudaFloat, typename MatrixCudaInt >
void cuda_test_GetType()
{
    MatrixCudaFloat mtrxCudaFloat;
    MatrixCudaInt mtrxCudaInt;

    EXPECT_EQ( mtrxCudaFloat.getType(), TNL::String( "Matrices::CSR< float, Cuda >" ) );
    EXPECT_EQ( mtrxCudaInt.getType(), TNL::String( "Matrices::CSR< int, Cuda >" ) );
}

template< typename Matrix >
void test_SetDimensions()
{
    const int rows = 9;
    const int cols = 8;
    
    Matrix m;
    m.setDimensions( rows, cols );
    
    EXPECT_EQ( m.getRows(), 9);
    EXPECT_EQ( m.getColumns(), 8);
}

template< typename Matrix >
void test_SetCompressedRowLengths()
{
    const int rows = 10;
    const int cols = 11;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 3 );
    int value = 1;
    for( int i = 2; i < rows; i++ )
        rowLengths.setElement( i, value++ );
    
    m.setCompressedRowLengths( rowLengths );
    
    EXPECT_EQ( m.getRowLength( 0), 3 );
    EXPECT_EQ( m.getRowLength( 1), 3 );
    EXPECT_EQ( m.getRowLength( 2), 1 );
    EXPECT_EQ( m.getRowLength( 3), 2 );
    EXPECT_EQ( m.getRowLength( 4), 3 );
    EXPECT_EQ( m.getRowLength( 5), 4 );
    EXPECT_EQ( m.getRowLength( 6), 5 );
    EXPECT_EQ( m.getRowLength( 7), 6 );
    EXPECT_EQ( m.getRowLength( 8), 7 );
    EXPECT_EQ( m.getRowLength( 9), 8 );
}

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
    const int rows = 8;
    const int cols = 7;
    
    Matrix1 m1;
    m1.reset();
    m1.setDimensions( rows + 1, cols + 2 );
    
    Matrix2 m2;
    m2.reset();
    m2.setDimensions( rows, cols );
    
    m1.setLike( m2 );
    
    EXPECT_EQ( m1.getRows(), m2.getRows() );
    EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void test_Reset()
{
    const int rows = 5;
    const int cols = 4;
    
    Matrix m;
    m.setDimensions( rows, cols );
    
    m.reset();
    
    EXPECT_EQ( m.getRows(), 0 );
    EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_SetElement()
{
    const int rows = 5;
    const int cols = 5;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 1 );
    m.setCompressedRowLengths( rowLengths );    
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        m.setElement( i, i, value++ );
    
    EXPECT_EQ( m.getElement( 0, 0 ), 1 );
    EXPECT_EQ( m.getElement( 1, 1 ), 2 );
    EXPECT_EQ( m.getElement( 2, 2 ), 3 );
    EXPECT_EQ( m.getElement( 3, 3 ), 4 );
    EXPECT_EQ( m.getElement( 4, 4 ), 5 );
}

TEST( SparseMatrixTest, CSR_GetTypeTest_Host )
{
   host_test_GetType< CSR_host_float, CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_GetTypeTest_Cuda )
{
   cuda_test_GetType< CSR_cuda_float, CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_SetDimensionsTest_Host )
{
   test_SetDimensions< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_SetDimensionsTest_Cuda )
{
   test_SetDimensions< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_setCompressedRowLengthsTest_Host )
{
   test_SetCompressedRowLengths< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_setCompressedRowLengthsTest_Cuda )
{
   test_SetCompressedRowLengths< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_setLikeTest_Host )
{
   test_SetLike< CSR_host_int, CSR_host_float >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_setLikeTest_Cuda )
{
   test_SetLike< CSR_cuda_int, CSR_cuda_float >();
}
#endif

TEST( SparseMatrixTest, CSR_resetTest_Host )
{
    test_Reset< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_resetTest_Cuda )
{
    test_Reset< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_setElementTest_Host )
{
    test_SetElement< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_setElementTest_Cuda )
{
    test_SetElement< CSR_cuda_int >();
}
#endif

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

