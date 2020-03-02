/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <iostream>
#include <sstream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template< typename MatrixHostFloat, typename MatrixHostInt >
void host_test_GetType()
{
   bool testRan = false;
   EXPECT_TRUE( testRan );
   std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
   std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
}

template< typename MatrixCudaFloat, typename MatrixCudaInt >
void cuda_test_GetType()
{
   bool testRan = false;
   EXPECT_TRUE( testRan );
   std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
   std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
}

template< typename Matrix >
void test_SetDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 9;
   const IndexType cols = 8;

   Matrix m;
   m.setDimensions( rows, cols );

   EXPECT_EQ( m.getRows(), 9 );
   EXPECT_EQ( m.getColumns(), 8 );
}

template< typename Matrix >
void test_SetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 10;
   const IndexType cols = 11;

   Matrix m( rows, cols );
   typename Matrix::CompressedRowLengthsVector rowLengths;
   rowLengths.setSize( rows );
   rowLengths = 3;

   IndexType rowLength = 1;
   for( IndexType i = 2; i < rows; i++ )
      rowLengths.setElement( i, rowLength++ );

   m.setCompressedRowLengths( rowLengths );

   // Insert values into the rows.
   RealType value = 1;

   for( IndexType i = 0; i < 3; i++ )      // 0th row
      m.setElement( 0, i, value++ );

   for( IndexType i = 0; i < 3; i++ )      // 1st row
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 1; i++ )      // 2nd row
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )      // 3rd row
      m.setElement( 3, i, value++ );

   for( IndexType i = 0; i < 3; i++ )      // 4th row
      m.setElement( 4, i, value++ );

   for( IndexType i = 0; i < 4; i++ )      // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 5; i++ )      // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 6; i++ )      // 7th row
      m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )      // 8th row
      m.setElement( 8, i, value++ );

   for( IndexType i = 0; i < 8; i++ )      // 9th row
      m.setElement( 9, i, value++ );

   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::CompressedRowLengthsVector correctRowLengths{ 3, 3, 1, 2, 3, 4, 5, 6, 7, 8 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

   Matrix1 m1( rows + 1, cols + 2 );
   Matrix2 m2( rows, cols );

   m1.setLike( m2 );

   EXPECT_EQ( m1.getRows(), m2.getRows() );
   EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void test_GetNumberOfNonzeroMatrixElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::CompressedRowLengthsVector rowLengths{ 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setCompressedRowLengths( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 0, 2 * i, value++ );

   for( IndexType i = 0; i < 3; i++ )
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 8; i++ )
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )
      m.setElement( 3, i, value++ );

   for( IndexType i = 4; i < 8; i++ )
      m.setElement( i, 0, value++ );

   for( IndexType j = 8; j < rows; j++)
      for( IndexType i = 0; i < cols; i++ )
         m.setElement( j, i, value++ );

   EXPECT_EQ( m.getNumberOfNonzeroMatrixElements(), 41 );
}

template< typename Matrix >
void test_Reset()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 sparse matrix:
    *
    *    /  0  0  0  0 \
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    \  0  0  0  0 /
    */

   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m( rows, cols );
   m.reset();

   EXPECT_EQ( m.getRows(), 0 );
   EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_GetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::CompressedRowLengthsVector rowLengths{ 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setCompressedRowLengths( rowLengths );

   auto matrixView = m.getView();
   auto f = [=] __cuda_callable__ ( const IndexType rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      RealType val;
      switch( rowIdx )
      {
         case 0:
            val = 1;
            for( IndexType i = 0; i < 4; i++ )
               row.setElement( i, 2 * i, val++ );
            break;
         case 1:
            val = 5;
            for( IndexType i = 0; i < 3; i++ )
               row.setElement( i, i, val++ );
            break;
         case 2:
            val = 8;
            for( IndexType i = 0; i < 8; i++ )
               row.setElement( i, i, val++ );
            break;
         case 3:
            val = 16;
            for( IndexType i = 0; i < 2; i++ )
               row.setElement( i, i, val++ );
            break;
         case 4:
            row.setElement( 0, 0, 18 );
            break;
         case 5:
            row.setElement( 0, 0, 19 );
            break;
         case 6:
            row.setElement( 0, 0, 20 );
            break;
         case 7:
            row.setElement( 0, 0, 21 );
            break;
         case 8:
            val = 22;
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, val++ );
            break;
         case 9:
            val = 32;
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, val++ );
            break;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, rows, f );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  2 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  3 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  4 );
   EXPECT_EQ( m.getElement( 0, 7 ),  0 );
   EXPECT_EQ( m.getElement( 0, 8 ),  0 );
   EXPECT_EQ( m.getElement( 0, 9 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  5 );
   EXPECT_EQ( m.getElement( 1, 1 ),  6 );
   EXPECT_EQ( m.getElement( 1, 2 ),  7 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );
   EXPECT_EQ( m.getElement( 1, 7 ),  0 );
   EXPECT_EQ( m.getElement( 1, 8 ),  0 );
   EXPECT_EQ( m.getElement( 1, 9 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  8 );
   EXPECT_EQ( m.getElement( 2, 1 ),  9 );
   EXPECT_EQ( m.getElement( 2, 2 ), 10 );
   EXPECT_EQ( m.getElement( 2, 3 ), 11 );
   EXPECT_EQ( m.getElement( 2, 4 ), 12 );
   EXPECT_EQ( m.getElement( 2, 5 ), 13 );
   EXPECT_EQ( m.getElement( 2, 6 ), 14 );
   EXPECT_EQ( m.getElement( 2, 7 ), 15 );
   EXPECT_EQ( m.getElement( 2, 8 ),  0 );
   EXPECT_EQ( m.getElement( 2, 9 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );
   EXPECT_EQ( m.getElement( 3, 6 ),  0 );
   EXPECT_EQ( m.getElement( 3, 7 ),  0 );
   EXPECT_EQ( m.getElement( 3, 8 ),  0 );
   EXPECT_EQ( m.getElement( 3, 9 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 18 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ),  0 );
   EXPECT_EQ( m.getElement( 4, 7 ),  0 );
   EXPECT_EQ( m.getElement( 4, 8 ),  0 );
   EXPECT_EQ( m.getElement( 4, 9 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );
   EXPECT_EQ( m.getElement( 5, 6 ),  0 );
   EXPECT_EQ( m.getElement( 5, 7 ),  0 );
   EXPECT_EQ( m.getElement( 5, 8 ),  0 );
   EXPECT_EQ( m.getElement( 5, 9 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 20 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
   EXPECT_EQ( m.getElement( 6, 6 ),  0 );
   EXPECT_EQ( m.getElement( 6, 7 ),  0 );
   EXPECT_EQ( m.getElement( 6, 8 ),  0 );
   EXPECT_EQ( m.getElement( 6, 9 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 21 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );
   EXPECT_EQ( m.getElement( 7, 6 ),  0 );
   EXPECT_EQ( m.getElement( 7, 7 ),  0 );
   EXPECT_EQ( m.getElement( 7, 8 ),  0 );
   EXPECT_EQ( m.getElement( 7, 9 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 22 );
   EXPECT_EQ( m.getElement( 8, 1 ), 23 );
   EXPECT_EQ( m.getElement( 8, 2 ), 24 );
   EXPECT_EQ( m.getElement( 8, 3 ), 25 );
   EXPECT_EQ( m.getElement( 8, 4 ), 26 );
   EXPECT_EQ( m.getElement( 8, 5 ), 27 );
   EXPECT_EQ( m.getElement( 8, 6 ), 28 );
   EXPECT_EQ( m.getElement( 8, 7 ), 29 );
   EXPECT_EQ( m.getElement( 8, 8 ), 30 );
   EXPECT_EQ( m.getElement( 8, 9 ), 31 );

   EXPECT_EQ( m.getElement( 9, 0 ), 32 );
   EXPECT_EQ( m.getElement( 9, 1 ), 33 );
   EXPECT_EQ( m.getElement( 9, 2 ), 34 );
   EXPECT_EQ( m.getElement( 9, 3 ), 35 );
   EXPECT_EQ( m.getElement( 9, 4 ), 36 );
   EXPECT_EQ( m.getElement( 9, 5 ), 37 );
   EXPECT_EQ( m.getElement( 9, 6 ), 38 );
   EXPECT_EQ( m.getElement( 9, 7 ), 39 );
   EXPECT_EQ( m.getElement( 9, 8 ), 40 );
   EXPECT_EQ( m.getElement( 9, 9 ), 41 );
}


template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m;
   m.reset();

   m.setDimensions( rows, cols );

   typename Matrix::CompressedRowLengthsVector rowLengths { 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setCompressedRowLengths( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 0, 2 * i, value++ );

   for( IndexType i = 0; i < 3; i++ )
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 8; i++ )
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )
      m.setElement( 3, i, value++ );

   for( IndexType i = 4; i < 8; i++ )
      m.setElement( i, 0, value++ );

   for( IndexType j = 8; j < rows; j++)
      for( IndexType i = 0; i < cols; i++ )
         m.setElement( j, i, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  2 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  3 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  4 );
   EXPECT_EQ( m.getElement( 0, 7 ),  0 );
   EXPECT_EQ( m.getElement( 0, 8 ),  0 );
   EXPECT_EQ( m.getElement( 0, 9 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  5 );
   EXPECT_EQ( m.getElement( 1, 1 ),  6 );
   EXPECT_EQ( m.getElement( 1, 2 ),  7 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );
   EXPECT_EQ( m.getElement( 1, 7 ),  0 );
   EXPECT_EQ( m.getElement( 1, 8 ),  0 );
   EXPECT_EQ( m.getElement( 1, 9 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  8 );
   EXPECT_EQ( m.getElement( 2, 1 ),  9 );
   EXPECT_EQ( m.getElement( 2, 2 ), 10 );
   EXPECT_EQ( m.getElement( 2, 3 ), 11 );
   EXPECT_EQ( m.getElement( 2, 4 ), 12 );
   EXPECT_EQ( m.getElement( 2, 5 ), 13 );
   EXPECT_EQ( m.getElement( 2, 6 ), 14 );
   EXPECT_EQ( m.getElement( 2, 7 ), 15 );
   EXPECT_EQ( m.getElement( 2, 8 ),  0 );
   EXPECT_EQ( m.getElement( 2, 9 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );
   EXPECT_EQ( m.getElement( 3, 6 ),  0 );
   EXPECT_EQ( m.getElement( 3, 7 ),  0 );
   EXPECT_EQ( m.getElement( 3, 8 ),  0 );
   EXPECT_EQ( m.getElement( 3, 9 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 18 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ),  0 );
   EXPECT_EQ( m.getElement( 4, 7 ),  0 );
   EXPECT_EQ( m.getElement( 4, 8 ),  0 );
   EXPECT_EQ( m.getElement( 4, 9 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );
   EXPECT_EQ( m.getElement( 5, 6 ),  0 );
   EXPECT_EQ( m.getElement( 5, 7 ),  0 );
   EXPECT_EQ( m.getElement( 5, 8 ),  0 );
   EXPECT_EQ( m.getElement( 5, 9 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 20 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
   EXPECT_EQ( m.getElement( 6, 6 ),  0 );
   EXPECT_EQ( m.getElement( 6, 7 ),  0 );
   EXPECT_EQ( m.getElement( 6, 8 ),  0 );
   EXPECT_EQ( m.getElement( 6, 9 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 21 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );
   EXPECT_EQ( m.getElement( 7, 6 ),  0 );
   EXPECT_EQ( m.getElement( 7, 7 ),  0 );
   EXPECT_EQ( m.getElement( 7, 8 ),  0 );
   EXPECT_EQ( m.getElement( 7, 9 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 22 );
   EXPECT_EQ( m.getElement( 8, 1 ), 23 );
   EXPECT_EQ( m.getElement( 8, 2 ), 24 );
   EXPECT_EQ( m.getElement( 8, 3 ), 25 );
   EXPECT_EQ( m.getElement( 8, 4 ), 26 );
   EXPECT_EQ( m.getElement( 8, 5 ), 27 );
   EXPECT_EQ( m.getElement( 8, 6 ), 28 );
   EXPECT_EQ( m.getElement( 8, 7 ), 29 );
   EXPECT_EQ( m.getElement( 8, 8 ), 30 );
   EXPECT_EQ( m.getElement( 8, 9 ), 31 );

   EXPECT_EQ( m.getElement( 9, 0 ), 32 );
   EXPECT_EQ( m.getElement( 9, 1 ), 33 );
   EXPECT_EQ( m.getElement( 9, 2 ), 34 );
   EXPECT_EQ( m.getElement( 9, 3 ), 35 );
   EXPECT_EQ( m.getElement( 9, 4 ), 36 );
   EXPECT_EQ( m.getElement( 9, 5 ), 37 );
   EXPECT_EQ( m.getElement( 9, 6 ), 38 );
   EXPECT_EQ( m.getElement( 9, 7 ), 39 );
   EXPECT_EQ( m.getElement( 9, 8 ), 40 );
   EXPECT_EQ( m.getElement( 9, 9 ), 41 );
}

template< typename Matrix >
void test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  3  0  0 \
    *    |  0  4  5  6  0 |
    *    |  0  0  7  8  9 |
    *    | 10  0  0  0  0 |
    *    |  0 11  0  0  0 |
    *    \  0  0  0 12  0 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );
   typename Matrix::CompressedRowLengthsVector rowLengths( rows );
   rowLengths = 3;
   m.setCompressedRowLengths( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < cols - 2; i++ )     // 0th row
      m.setElement( 0, i, value++ );

   for( IndexType i = 1; i < cols - 1; i++ )     // 1st row
      m.setElement( 1, i, value++ );

   for( IndexType i = 2; i < cols; i++ )         // 2nd row
      m.setElement( 2, i, value++ );

   m.setElement( 3, 0, value++ );      // 3rd row

   m.setElement( 4, 1, value++ );      // 4th row

   m.setElement( 5, 3, value++ );      // 5th row


   // Check the set elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m.getElement( 2, 4 ),  9 );

   EXPECT_EQ( m.getElement( 3, 0 ), 10 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 11 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ), 12 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *    /  3  6  9  0  0 \
    *    |  0 12 15 18  0 |
    *    |  0  0 21 24 27 |
    *    | 30 11 12  0  0 |
    *    |  0 35 14 15  0 |
    *    \  0  0 16 41 18 /
    */

   RealType newValue = 1;
   for( IndexType i = 0; i < cols - 2; i++ )         // 0th row
      m.addElement( 0, i, newValue++, 2.0 );

   for( IndexType i = 1; i < cols - 1; i++ )         // 1st row
      m.addElement( 1, i, newValue++, 2.0 );

   for( IndexType i = 2; i < cols; i++ )             // 2nd row
      m.addElement( 2, i, newValue++, 2.0 );

   for( IndexType i = 0; i < cols - 2; i++ )         // 3rd row
      m.addElement( 3, i, newValue++, 2.0 );

   for( IndexType i = 1; i < cols - 1; i++ )         // 4th row
      m.addElement( 4, i, newValue++, 2.0 );

   for( IndexType i = 2; i < cols; i++ )             // 5th row
      m.addElement( 5, i, newValue++, 2.0 );


   EXPECT_EQ( m.getElement( 0, 0 ),  3 );
   EXPECT_EQ( m.getElement( 0, 1 ),  6 );
   EXPECT_EQ( m.getElement( 0, 2 ),  9 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ), 12 );
   EXPECT_EQ( m.getElement( 1, 2 ), 15 );
   EXPECT_EQ( m.getElement( 1, 3 ), 18 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 21 );
   EXPECT_EQ( m.getElement( 2, 3 ), 24 );
   EXPECT_EQ( m.getElement( 2, 4 ), 27 );

   EXPECT_EQ( m.getElement( 3, 0 ), 30 );
   EXPECT_EQ( m.getElement( 3, 1 ), 11 );
   EXPECT_EQ( m.getElement( 3, 2 ), 12 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 35 );
   EXPECT_EQ( m.getElement( 4, 2 ), 14 );
   EXPECT_EQ( m.getElement( 4, 3 ), 15 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ), 16 );
   EXPECT_EQ( m.getElement( 5, 3 ), 41 );
   EXPECT_EQ( m.getElement( 5, 4 ), 18 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  0  0  0 \
    *    |  0  2  0  3 |
    *    |  0  4  0  0 |
    *    \  0  0  5  0 /
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1;
   m_1.reset();
   m_1.setDimensions( m_rows_1, m_cols_1 );
   typename Matrix::CompressedRowLengthsVector rowLengths_1;
   rowLengths_1.setSize( m_rows_1 );
   rowLengths_1.setElement( 0, 1 );
   rowLengths_1.setElement( 1, 2 );
   rowLengths_1.setElement( 2, 1 );
   rowLengths_1.setElement( 3, 1 );
   m_1.setCompressedRowLengths( rowLengths_1 );

   RealType value_1 = 1;
   m_1.setElement( 0, 0, value_1++ );      // 0th row

   m_1.setElement( 1, 1, value_1++ );      // 1st row
   m_1.setElement( 1, 3, value_1++ );

   m_1.setElement( 2, 1, value_1++ );      // 2nd row

   m_1.setElement( 3, 2, value_1++ );      // 3rd row

   VectorType inVector_1;
   inVector_1.setSize( m_cols_1 );
   for( IndexType i = 0; i < inVector_1.getSize(); i++ )
       inVector_1.setElement( i, 2 );

   VectorType outVector_1;
   outVector_1.setSize( m_rows_1 );
   for( IndexType j = 0; j < outVector_1.getSize(); j++ )
       outVector_1.setElement( j, 0 );


   m_1.vectorProduct( inVector_1, outVector_1 );


   EXPECT_EQ( outVector_1.getElement( 0 ),  2 );
   EXPECT_EQ( outVector_1.getElement( 1 ), 10 );
   EXPECT_EQ( outVector_1.getElement( 2 ),  8 );
   EXPECT_EQ( outVector_1.getElement( 3 ), 10 );

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  0  0  4 |
    *    |  5  6  7  0 |
    *    \  0  8  0  0 /
    */

   const IndexType m_rows_2 = 4;
   const IndexType m_cols_2 = 4;

   Matrix m_2( m_rows_2, m_cols_2 );
   typename Matrix::CompressedRowLengthsVector rowLengths_2{ 3, 1, 3, 1 };
   /*rowLengths_2 = 3;
   rowLengths_2.setElement( 1, 1 );
   rowLengths_2.setElement( 3, 1 );*/
   m_2.setCompressedRowLengths( rowLengths_2 );

   RealType value_2 = 1;
   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m_2.setElement( 0, i, value_2++ );

   m_2.setElement( 1, 3, value_2++ );      // 1st row

   for( IndexType i = 0; i < 3; i++ )   // 2nd row
      m_2.setElement( 2, i, value_2++ );

   for( IndexType i = 1; i < 2; i++ )       // 3rd row
      m_2.setElement( 3, i, value_2++ );

   VectorType inVector_2;
   inVector_2.setSize( m_cols_2 );
   for( IndexType i = 0; i < inVector_2.getSize(); i++ )
      inVector_2.setElement( i, 2 );

   VectorType outVector_2;
   outVector_2.setSize( m_rows_2 );
   for( IndexType j = 0; j < outVector_2.getSize(); j++ )
      outVector_2.setElement( j, 0 );

   m_2.vectorProduct( inVector_2, outVector_2 );

   EXPECT_EQ( outVector_2.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_2.getElement( 1 ),  8 );
   EXPECT_EQ( outVector_2.getElement( 2 ), 36 );
   EXPECT_EQ( outVector_2.getElement( 3 ), 16 );

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  4  5  6 |
    *    |  7  8  9  0 |
    *    \  0 10 11 12 /
    */

   const IndexType m_rows_3 = 4;
   const IndexType m_cols_3 = 4;

   Matrix m_3( m_rows_3, m_cols_3 );
   typename Matrix::CompressedRowLengthsVector rowLengths_3{ 3, 3, 3, 3 };
   m_3.setCompressedRowLengths( rowLengths_3 );

   RealType value_3 = 1;
   for( IndexType i = 0; i < 3; i++ )          // 0th row
      m_3.setElement( 0, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )
      m_3.setElement( 1, i, value_3++ );      // 1st row

   for( IndexType i = 0; i < 3; i++ )          // 2nd row
      m_3.setElement( 2, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )          // 3rd row
      m_3.setElement( 3, i, value_3++ );

   VectorType inVector_3;
   inVector_3.setSize( m_cols_3 );
   for( IndexType i = 0; i < inVector_3.getSize(); i++ )
      inVector_3.setElement( i, 2 );

   VectorType outVector_3;
   outVector_3.setSize( m_rows_3 );
   for( IndexType j = 0; j < outVector_3.getSize(); j++ )
      outVector_3.setElement( j, 0 );

   m_3.vectorProduct( inVector_3, outVector_3 );

   EXPECT_EQ( outVector_3.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_3.getElement( 1 ), 30 );
   EXPECT_EQ( outVector_3.getElement( 2 ), 48 );
   EXPECT_EQ( outVector_3.getElement( 3 ), 66 );

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  0  4  0  0 \
    *    |  0  5  6  7  8  0  0  0 |
    *    |  9 10 11 12 13  0  0  0 |
    *    |  0 14 15 16 17  0  0  0 |
    *    |  0  0 18 19 20 21  0  0 |
    *    |  0  0  0 22 23 24 25  0 |
    *    | 26 27 28 29 30  0  0  0 |
    *    \ 31 32 33 34 35  0  0  0 /
    */

   const IndexType m_rows_4 = 8;
   const IndexType m_cols_4 = 8;

   Matrix m_4( m_rows_4, m_cols_4 );
   typename Matrix::CompressedRowLengthsVector rowLengths_4{ 4, 4, 5, 4, 4, 4, 5, 5 };
   /*rowLengths_4.setSize( m_rows_4 );
   rowLengths_4.setValue( 4 );
   rowLengths_4.setElement( 2, 5 );
   rowLengths_4.setElement( 6, 5 );
   rowLengths_4.setElement( 7, 5 );*/
   m_4.setCompressedRowLengths( rowLengths_4 );

   RealType value_4 = 1;
   for( IndexType i = 0; i < 3; i++ )       // 0th row
      m_4.setElement( 0, i, value_4++ );

   m_4.setElement( 0, 5, value_4++ );

   for( IndexType i = 1; i < 5; i++ )       // 1st row
      m_4.setElement( 1, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 2nd row
      m_4.setElement( 2, i, value_4++ );

   for( IndexType i = 1; i < 5; i++ )       // 3rd row
      m_4.setElement( 3, i, value_4++ );

   for( IndexType i = 2; i < 6; i++ )       // 4th row
      m_4.setElement( 4, i, value_4++ );

   for( IndexType i = 3; i < 7; i++ )       // 5th row
      m_4.setElement( 5, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 6th row
      m_4.setElement( 6, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 7th row
      m_4.setElement( 7, i, value_4++ );

   VectorType inVector_4;
   inVector_4.setSize( m_cols_4 );
   for( IndexType i = 0; i < inVector_4.getSize(); i++ )
      inVector_4.setElement( i, 2 );

   VectorType outVector_4;
   outVector_4.setSize( m_rows_4 );
   for( IndexType j = 0; j < outVector_4.getSize(); j++ )
      outVector_4.setElement( j, 0 );

   m_4.vectorProduct( inVector_4, outVector_4 );

   EXPECT_EQ( outVector_4.getElement( 0 ),  20 );
   EXPECT_EQ( outVector_4.getElement( 1 ),  52 );
   EXPECT_EQ( outVector_4.getElement( 2 ), 110 );
   EXPECT_EQ( outVector_4.getElement( 3 ), 124 );
   EXPECT_EQ( outVector_4.getElement( 4 ), 156 );
   EXPECT_EQ( outVector_4.getElement( 5 ), 188 );
   EXPECT_EQ( outVector_4.getElement( 6 ), 280 );
   EXPECT_EQ( outVector_4.getElement( 7 ), 330 );


   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  4  5  0  1 \   6
    *    |  0  6  0  7  0  0  0  1 |   3
    *    |  0  8  9  0 10  0  0  1 |   4
    *    |  0 11 12 13 14  0  0  1 |   5
    *    |  0 15  0  0  0  0  0  1 |   2
    *    |  0 16 17 18 19 20 21  1 |   7
    *    | 22 23 24 25 26 27 28  1 |   8
    *    \ 29 30 31 32 33 34 35 36 /   8
    */

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   Matrix m_5( m_rows_5, m_cols_5 );
   typename Matrix::CompressedRowLengthsVector rowLengths_5{ 6, 3, 4, 5, 2, 7, 8, 8 };
   m_5.setCompressedRowLengths( rowLengths_5 );

   RealType value_5 = 1;
   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m_5.setElement( 0, i, value_5++ );

   m_5.setElement( 0, 4, value_5++ );           // 0th row
   m_5.setElement( 0, 5, value_5++ );

   m_5.setElement( 1, 1, value_5++ );           // 1st row
   m_5.setElement( 1, 3, value_5++ );

   for( IndexType i = 1; i < 3; i++ )            // 2nd row
      m_5.setElement( 2, i, value_5++ );

   m_5.setElement( 2, 4, value_5++ );           // 2nd row

   for( IndexType i = 1; i < 5; i++ )            // 3rd row
      m_5.setElement( 3, i, value_5++ );

   m_5.setElement( 4, 1, value_5++ );           // 4th row

   for( IndexType i = 1; i < 7; i++ )            // 5th row
      m_5.setElement( 5, i, value_5++ );

   for( IndexType i = 0; i < 7; i++ )            // 6th row
      m_5.setElement( 6, i, value_5++ );

   for( IndexType i = 0; i < 8; i++ )            // 7th row
      m_5.setElement( 7, i, value_5++ );

   for( IndexType i = 0; i < 7; i++ )            // 1s at the end of rows
      m_5.setElement( i, 7, 1);

   VectorType inVector_5;
   inVector_5.setSize( m_cols_5 );
   for( IndexType i = 0; i < inVector_5.getSize(); i++ )
       inVector_5.setElement( i, 2 );

   VectorType outVector_5;
   outVector_5.setSize( m_rows_5 );
   for( IndexType j = 0; j < outVector_5.getSize(); j++ )
       outVector_5.setElement( j, 0 );

   m_5.vectorProduct( inVector_5, outVector_5 );

   EXPECT_EQ( outVector_5.getElement( 0 ),  32 );
   EXPECT_EQ( outVector_5.getElement( 1 ),  28 );
   EXPECT_EQ( outVector_5.getElement( 2 ),  56 );
   EXPECT_EQ( outVector_5.getElement( 3 ), 102 );
   EXPECT_EQ( outVector_5.getElement( 4 ),  32 );
   EXPECT_EQ( outVector_5.getElement( 5 ), 224 );
   EXPECT_EQ( outVector_5.getElement( 6 ), 352 );
   EXPECT_EQ( outVector_5.getElement( 7 ), 520 );
}

template< typename Matrix >
void test_RowsReduction()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  4  5  0  1 \   6
    *    |  0  6  0  7  0  0  0  1 |   3
    *    |  0  8  9  0 10  0  0  1 |   4
    *    |  0 11 12 13 14  0  0  1 |   5
    *    |  0 15  0  0  0  0  0  1 |   2
    *    |  0 16 17 18 19 20 21  1 |   7
    *    | 22 23 24 25 26 27 28  1 |   8
    *    \ 29 30 31 32 33 34 35 36 /   8
    */

   const IndexType rows = 8;
   const IndexType cols = 8;

   Matrix m;
   m.setDimensions( rows, cols );
   typename Matrix::RowsCapacitiesType rowsCapacities{ 6, 3, 4, 5, 2, 7, 8, 8 };
   m.setCompressedRowLengths( rowsCapacities );

   RealType value = 1;
   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m.setElement( 0, i, value++ );

   m.setElement( 0, 4, value++ );       // 0th row
   m.setElement( 0, 5, value++ );

   m.setElement( 1, 1, value++ );       // 1st row
   m.setElement( 1, 3, value++ );

   for( IndexType i = 1; i < 3; i++ )   // 2nd row
      m.setElement( 2, i, value++ );

   m.setElement( 2, 4, value++ );       // 2nd row

   for( IndexType i = 1; i < 5; i++ )   // 3rd row
      m.setElement( 3, i, value++ );

   m.setElement( 4, 1, value++ );       // 4th row

   for( IndexType i = 1; i < 7; i++ )   // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 7; i++ )   // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 8; i++ )   // 7th row
       m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )   // 1s at the end of rows
      m.setElement( i, 7, 1);

   ////
   // Compute number of non-zero elements in rows.
   typename Matrix::RowsCapacitiesType rowLengths( rows );
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, IndexType globalIdx, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType a ) {
      aux += a;
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   m.allRowsReduction( fetch, reduce, keep, 0 );
   EXPECT_EQ( rowsCapacities, rowLengths );
   m.getCompressedRowLengths( rowLengths );
   EXPECT_EQ( rowsCapacities, rowLengths );

   ////
   // Compute max norm
   TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( rows );
   auto rowSums_view = rowSums.getView();
   auto max_fetch = [] __cuda_callable__ ( IndexType row, IndexType column, IndexType globalIdx, const RealType& value ) -> IndexType {
      return abs( value );
   };
   auto max_reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType a ) {
      aux += a;
   };
   auto max_keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowSums_view[ rowIdx ] = value;
   };
   m.allRowsReduction( max_fetch, max_reduce, max_keep, 0 );
   const RealType maxNorm = TNL::max( rowSums );
   EXPECT_EQ( maxNorm, 260 ) ; // 29+30+31+32+33+34+35+36
}

template< typename Matrix >
void test_PerformSORIteration()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  4  1  0  0 \
    *    |  1  4  1  0 |
    *    |  0  1  4  1 |
    *    \  0  0  1  4 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix m( m_rows, m_cols );
   typename Matrix::CompressedRowLengthsVector rowLengths( m_rows );
   rowLengths = 3;
   m.setCompressedRowLengths( rowLengths );

   m.setElement( 0, 0, 4.0 );        // 0th row
   m.setElement( 0, 1, 1.0);

   m.setElement( 1, 0, 1.0 );        // 1st row
   m.setElement( 1, 1, 4.0 );
   m.setElement( 1, 2, 1.0 );

   m.setElement( 2, 1, 1.0 );        // 2nd row
   m.setElement( 2, 2, 4.0 );
   m.setElement( 2, 3, 1.0 );

   m.setElement( 3, 2, 1.0 );        // 3rd row
   m.setElement( 3, 3, 4.0 );

   RealType bVector [ 4 ] = { 1, 1, 1, 1 };
   RealType xVector [ 4 ] = { 1, 1, 1, 1 };

   IndexType row = 0;
   RealType omega = 1;

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 1.0 );
   EXPECT_EQ( xVector[ 2 ], 1.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 1.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 0.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 0.0 );
   EXPECT_EQ( xVector[ 3 ], 0.25 );
}

template< typename Matrix >
void test_SaveAndLoad( const char* filename )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  4  0  5 |
    *    |  6  7  8  0 |
    *    \  0  9 10 11 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix savedMatrix( m_rows, m_cols );
   typename Matrix::CompressedRowLengthsVector rowLengths( m_rows );
   rowLengths = 3;
   savedMatrix.setCompressedRowLengths( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
      savedMatrix.setElement( 0, i, value++ );

   savedMatrix.setElement( 1, 1, value++ );
   savedMatrix.setElement( 1, 3, value++ );      // 1st row

   for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
      savedMatrix.setElement( 2, i, value++ );

   for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
      savedMatrix.setElement( 3, i, value++ );

   ASSERT_NO_THROW( savedMatrix.save( filename ) );

   Matrix loadedMatrix( m_rows, m_cols );
   typename Matrix::CompressedRowLengthsVector rowLengths2( m_rows );
   rowLengths2 = 3;
   loadedMatrix.setCompressedRowLengths( rowLengths2 );

   ASSERT_NO_THROW( loadedMatrix.load( filename ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ), loadedMatrix.getElement( 0, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ), loadedMatrix.getElement( 0, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ), loadedMatrix.getElement( 0, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ), loadedMatrix.getElement( 0, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ), loadedMatrix.getElement( 1, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ), loadedMatrix.getElement( 1, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ), loadedMatrix.getElement( 1, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ), loadedMatrix.getElement( 1, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ), loadedMatrix.getElement( 2, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), loadedMatrix.getElement( 2, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), loadedMatrix.getElement( 2, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), loadedMatrix.getElement( 2, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ), loadedMatrix.getElement( 3, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ), loadedMatrix.getElement( 3, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), loadedMatrix.getElement( 3, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), loadedMatrix.getElement( 3, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ),  2 );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ),  3 );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  4 );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  5 );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  6 );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ),  7 );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ),  8 );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ),  9 );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), 10 );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), 11 );

   EXPECT_EQ( std::remove( filename ), 0 );
}

template< typename Matrix >
void test_Print()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  0  0  4 |
    *    |  5  6  7  0 |
    *    |  0  8  9 10 |
    *    \  0  0 11 12 /
    */

   const IndexType m_rows = 5;
   const IndexType m_cols = 4;

   Matrix m( m_rows, m_cols );
   typename Matrix::CompressedRowLengthsVector rowLengths( m_rows );
   rowLengths = 3;
   m.setCompressedRowLengths( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
      m.setElement( 0, i, value++ );

   m.setElement( 1, 3, value++ );                // 1st row

   for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
      m.setElement( 2, i, value++ );

   for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
      m.setElement( 3, i, value++ );

   for( IndexType i = 2; i < m_cols; i++ )       // 4th row
      m.setElement( 4, i, value++ );

   std::stringstream printed;
   std::stringstream couted;

   //change the underlying buffer and save the old buffer
   auto old_buf = std::cout.rdbuf(printed.rdbuf());

   m.print( std::cout ); //all the std::cout goes to ss

   std::cout.rdbuf(old_buf); //reset

   couted << "Row: 0 ->  Col:0->1	 Col:1->2	 Col:2->3\t\n"
             "Row: 1 ->  Col:3->4\t\n"
             "Row: 2 ->  Col:0->5	 Col:1->6	 Col:2->7\t\n"
             "Row: 3 ->  Col:1->8	 Col:2->9	 Col:3->10\t\n"
             "Row: 4 ->  Col:2->11	 Col:3->12\t\n";

   EXPECT_EQ( printed.str(), couted.str() );
}

#endif
