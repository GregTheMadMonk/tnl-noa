/***************************************************************************
                          SparseMatrixCopyTest.h -  description
                             -------------------
    begin                : Jun 25, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/Legacy/CSR.h>
#include <TNL/Matrices/Legacy/Ellpack.h>
#include <TNL/Matrices/Legacy/SlicedEllpack.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Containers/Segments/CSR.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Containers/Segments/SlicedEllpack.h>

using CSR_host = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;
using CSR_cuda = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;
using E_host = TNL::Matrices::Ellpack< int, TNL::Devices::Host, int >;
using E_cuda = TNL::Matrices::Ellpack< int, TNL::Devices::Cuda, int >;
using SE_host = TNL::Matrices::SlicedEllpack< int, TNL::Devices::Host, int, 2 >;
using SE_cuda = TNL::Matrices::SlicedEllpack< int, TNL::Devices::Cuda, int, 2 >;

/*template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = TNL::Containers::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

using CSR_host = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Containers::Segments::CSR >;
using CSR_cuda = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Containers::Segments::CSR >;
using E_host   = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, EllpackSegments >;
using E_cuda   = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, EllpackSegments >;
using SE_host  = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SlicedEllpackSegments >;
using SE_cuda  = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SlicedEllpackSegments >;*/


#ifdef HAVE_GTEST
#include <gtest/gtest.h>

/*
 * Sets up the following 10x6 sparse matrix:
 *
 *    /  1  2             \
 *    |           3  4  5 |
 *    |  6  7  8          |
 *    |     9 10 11 12 13 |
 *    | 14 15 16 17 18    |
 *    | 19 20             |
 *    | 21                |
 *    | 22                |
 *    | 23 24 25 26 27    |
 *    \                28 /
 */
template< typename Matrix >
void setupUnevenRowSizeMatrix( Matrix& m )
{
    const int rows = 10;
    const int cols = 6;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 5 );
    rowLengths.setElement( 0, 2 );
    rowLengths.setElement( 1,  3 );
    rowLengths.setElement( 2,  3 );
    rowLengths.setElement( 5,  2 );
    rowLengths.setElement( 6,  1 );
    rowLengths.setElement( 7,  1 );
    rowLengths.setElement( 9,  1 );
    m.setCompressedRowLengths( rowLengths );

    int value = 1;
    for( int i = 0; i < cols - 4; i++ )  // 0th row
        m.setElement( 0, i, value++ );

    for( int i = 3; i < cols; i++ )      // 1st row
        m.setElement( 1, i, value++ );

    for( int i = 0; i < cols - 3; i++ )  // 2nd row
        m.setElement( 2, i, value++ );

    for( int i = 1; i < cols; i++ )      // 3rd row
        m.setElement( 3, i, value++ );

    for( int i = 0; i < cols - 1; i++ )  // 4th row
        m.setElement( 4, i, value++ );

    for( int i = 0; i < cols - 4; i++ )  // 5th row
        m.setElement( 5, i, value++ );

    m.setElement( 6, 0, value++ );   // 6th row

    m.setElement( 7, 0, value++ );   // 7th row

    for( int i = 0; i < cols - 1; i++ )  // 8th row
        m.setElement( 8, i, value++ );

    m.setElement( 9, 5, value++ );   // 9th row
}

template< typename Matrix >
void checkUnevenRowSizeMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 10 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0);

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  3 );
   EXPECT_EQ( m.getElement( 1, 4 ),  4 );
   EXPECT_EQ( m.getElement( 1, 5 ),  5 );

   EXPECT_EQ( m.getElement( 2, 0 ),  6 );
   EXPECT_EQ( m.getElement( 2, 1 ),  7 );
   EXPECT_EQ( m.getElement( 2, 2 ),  8 );
   EXPECT_EQ( m.getElement( 2, 3 ),  0 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  9 );
   EXPECT_EQ( m.getElement( 3, 2 ), 10 );
   EXPECT_EQ( m.getElement( 3, 3 ), 11 );
   EXPECT_EQ( m.getElement( 3, 4 ), 12 );
   EXPECT_EQ( m.getElement( 3, 5 ), 13 );

   EXPECT_EQ( m.getElement( 4, 0 ), 14 );
   EXPECT_EQ( m.getElement( 4, 1 ), 15 );
   EXPECT_EQ( m.getElement( 4, 2 ), 16 );
   EXPECT_EQ( m.getElement( 4, 3 ), 17 );
   EXPECT_EQ( m.getElement( 4, 4 ), 18 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ), 20 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 21 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 22 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 23 );
   EXPECT_EQ( m.getElement( 8, 1 ), 24 );
   EXPECT_EQ( m.getElement( 8, 2 ), 25 );
   EXPECT_EQ( m.getElement( 8, 3 ), 26 );
   EXPECT_EQ( m.getElement( 8, 4 ), 27 );
   EXPECT_EQ( m.getElement( 8, 5 ),  0 );

   EXPECT_EQ( m.getElement( 9, 0 ),  0 );
   EXPECT_EQ( m.getElement( 9, 1 ),  0 );
   EXPECT_EQ( m.getElement( 9, 2 ),  0 );
   EXPECT_EQ( m.getElement( 9, 3 ),  0 );
   EXPECT_EQ( m.getElement( 9, 4 ),  0 );
   EXPECT_EQ( m.getElement( 9, 5 ), 28 );
}

/*
 * Sets up the following 7x6 sparse matrix:
 *
 *    /              2  1 \
 *    |           5  4  3 |
 *    |        8  7  6    |
 *    |    11 10  9       |
 *    | 14 13 12          |
 *    | 16 15             |
 *    \ 17                /
 */
template< typename Matrix >
void setupAntiTriDiagMatrix( Matrix& m )
{
    const int rows = 7;
    const int cols = 6;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 3 );
    rowLengths.setElement( 0, 4);
    rowLengths.setElement( 1,  4 );
    m.setCompressedRowLengths( rowLengths );

    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = cols - 1; j > 2; j-- )
            if( j - i + 1 < cols && j - i + 1 >= 0 )
                m.setElement( i, j - i + 1, value++ );
}

template< typename Matrix >
void checkAntiTriDiagMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 7 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  0 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  2 );
   EXPECT_EQ( m.getElement( 0, 5 ),  1);

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  5 );
   EXPECT_EQ( m.getElement( 1, 4 ),  4 );
   EXPECT_EQ( m.getElement( 1, 5 ),  3 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  8 );
   EXPECT_EQ( m.getElement( 2, 3 ),  7 );
   EXPECT_EQ( m.getElement( 2, 4 ),  6 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ), 11 );
   EXPECT_EQ( m.getElement( 3, 2 ), 10 );
   EXPECT_EQ( m.getElement( 3, 3 ),  9 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 14 );
   EXPECT_EQ( m.getElement( 4, 1 ), 13 );
   EXPECT_EQ( m.getElement( 4, 2 ), 12 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 16 );
   EXPECT_EQ( m.getElement( 5, 1 ), 15 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 17 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
}

/*
 * Sets up the following 7x6 sparse matrix:
 *
 *    / 1  2             \
 *    | 3  4  5          |
 *    |    6  7  8       |
 *    |       9 10 11    |
 *    |         12 13 14 |
 *    |            15 16 |
 *    \               17 /
 */
template< typename Matrix >
void setupTriDiagMatrix( Matrix& m )
{
   const int rows = 7;
   const int cols = 6;
   m.reset();
   m.setDimensions( rows, cols );
   typename Matrix::CompressedRowLengthsVector rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setValue( 3 );
   rowLengths.setElement( 0 , 4 );
   rowLengths.setElement( 1,  4 );
   m.setCompressedRowLengths( rowLengths );

   int value = 1;
   for( int i = 0; i < rows; i++ )
      for( int j = 0; j < 3; j++ )
         if( i + j - 1 >= 0 && i + j - 1 < cols )
            m.setElement( i, i + j - 1, value++ );
}

template< typename Matrix >
void checkTriDiagMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 7 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  3 );
   EXPECT_EQ( m.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  6 );
   EXPECT_EQ( m.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  9 );
   EXPECT_EQ( m.getElement( 3, 3 ), 10 );
   EXPECT_EQ( m.getElement( 3, 4 ), 11 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 12 );
   EXPECT_EQ( m.getElement( 4, 4 ), 13 );
   EXPECT_EQ( m.getElement( 4, 5 ), 14 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 15 );
   EXPECT_EQ( m.getElement( 5, 5 ), 16 );

   EXPECT_EQ( m.getElement( 6, 0 ),  0 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 17 );
}

template< typename Matrix1, typename Matrix2 >
void testCopyAssignment()
{
   {
      SCOPED_TRACE("Tri Diagonal Matrix");

      Matrix1 triDiag1;
      setupTriDiagMatrix( triDiag1 );
      checkTriDiagMatrix( triDiag1 );

      Matrix2 triDiag2;
      triDiag2 = triDiag1;
      checkTriDiagMatrix( triDiag2 );
   }
   {
      SCOPED_TRACE("Anti Tri Diagonal Matrix");
      Matrix1 antiTriDiag1;
      setupAntiTriDiagMatrix( antiTriDiag1 );
      checkAntiTriDiagMatrix( antiTriDiag1 );

      Matrix2 antiTriDiag2;
      antiTriDiag2 = antiTriDiag1;
      checkAntiTriDiagMatrix( antiTriDiag2 );
   }
   {
      SCOPED_TRACE("Uneven Row Size Matrix");
      Matrix1 unevenRowSize1;
      setupUnevenRowSizeMatrix( unevenRowSize1 );
      checkUnevenRowSizeMatrix( unevenRowSize1 );

      Matrix2 unevenRowSize2;
      unevenRowSize2 = unevenRowSize1;
      checkUnevenRowSizeMatrix( unevenRowSize2 );
   }
}

template< typename Matrix1, typename Matrix2 >
void testConversion()
{

   {
        SCOPED_TRACE("Tri Diagonal Matrix");

        Matrix1 triDiag1;
        setupTriDiagMatrix( triDiag1 );
        checkTriDiagMatrix( triDiag1 );

        Matrix2 triDiag2;
        //TNL::Matrices::copySparseMatrix( triDiag2, triDiag1 );
        triDiag2 = triDiag1;
        checkTriDiagMatrix( triDiag2 );
   }

   {
        SCOPED_TRACE("Anti Tri Diagonal Matrix");

        Matrix1 antiTriDiag1;
        setupAntiTriDiagMatrix( antiTriDiag1 );
        checkAntiTriDiagMatrix( antiTriDiag1 );

        Matrix2 antiTriDiag2;
        //TNL::Matrices::copySparseMatrix( antiTriDiag2, antiTriDiag1 );
        antiTriDiag2 = antiTriDiag1;
        checkAntiTriDiagMatrix( antiTriDiag2 );
   }

   {
        SCOPED_TRACE("Uneven Row Size Matrix");
        Matrix1 unevenRowSize1;
        setupUnevenRowSizeMatrix( unevenRowSize1 );
        checkUnevenRowSizeMatrix( unevenRowSize1 );

        Matrix2 unevenRowSize2;
        //TNL::Matrices::copySparseMatrix( unevenRowSize2, unevenRowSize1 );
        unevenRowSize2 = unevenRowSize1;
        checkUnevenRowSizeMatrix( unevenRowSize2 );
   }
}

TEST( SparseMatrixCopyTest, CSR_HostToHost )
{
   testCopyAssignment< CSR_host, CSR_host >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixCopyTest, CSR_HostToCuda )
{
   testCopyAssignment< CSR_host, CSR_cuda >();
}

TEST( SparseMatrixCopyTest, CSR_CudaToHost )
{
   testCopyAssignment< CSR_cuda, CSR_host >();
}

TEST( SparseMatrixCopyTest, CSR_CudaToCuda )
{
   testCopyAssignment< CSR_cuda, CSR_cuda >();
}
#endif


TEST( SparseMatrixCopyTest, Ellpack_HostToHost )
{
   testCopyAssignment< E_host, E_host >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixCopyTest, Ellpack_HostToCuda )
{
   testCopyAssignment< E_host, E_cuda >();
}

TEST( SparseMatrixCopyTest, Ellpack_CudaToHost )
{
   testCopyAssignment< E_cuda, E_host >();
}

TEST( SparseMatrixCopyTest, Ellpack_CudaToCuda )
{
   testCopyAssignment< E_cuda, E_cuda >();
}
#endif


TEST( SparseMatrixCopyTest, SlicedEllpack_HostToHost )
{
   testCopyAssignment< SE_host, SE_host >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixCopyTest, SlicedEllpack_HostToCuda )
{
   testCopyAssignment< SE_host, SE_cuda >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_CudaToHost )
{
   testCopyAssignment< SE_cuda, SE_host >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_CudaToCuda )
{
   testCopyAssignment< SE_cuda, SE_cuda >();
}
#endif


// test conversion between formats
TEST( SparseMatrixCopyTest, CSR_to_Ellpack_host )
{
   testConversion< CSR_host, E_host >();
}

TEST( SparseMatrixCopyTest, Ellpack_to_CSR_host )
{
   testConversion< E_host, CSR_host >();
}

TEST( SparseMatrixCopyTest, CSR_to_SlicedEllpack_host )
{
   testConversion< CSR_host, SE_host >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_to_CSR_host )
{
   testConversion< SE_host, CSR_host >();
}

TEST( SparseMatrixCopyTest, Ellpack_to_SlicedEllpack_host )
{
   testConversion< E_host, SE_host >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_to_Ellpack_host )
{
   testConversion< SE_host, E_host >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixCopyTest, CSR_to_Ellpack_cuda )
{
   testConversion< CSR_cuda, E_cuda >();
}

TEST( SparseMatrixCopyTest, Ellpack_to_CSR_cuda )
{
   testConversion< E_cuda, CSR_cuda >();
}

TEST( SparseMatrixCopyTest, CSR_to_SlicedEllpack_cuda )
{
   testConversion< CSR_cuda, SE_cuda >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_to_CSR_cuda )
{
   testConversion< SE_cuda, CSR_cuda >();
}

TEST( SparseMatrixCopyTest, Ellpack_to_SlicedEllpack_cuda )
{
   testConversion< E_cuda, SE_cuda >();
}

TEST( SparseMatrixCopyTest, SlicedEllpack_to_Ellpack_cuda )
{
   testConversion< SE_cuda, E_cuda >();
}
#endif

#endif

#include "../../main.h"
