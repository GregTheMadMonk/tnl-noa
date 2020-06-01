/***************************************************************************
                          MultidiagonalMatrixTest.h -  description
                             -------------------
    begin                : Jan 8, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <sstream>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Containers/Array.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

using Multidiagonal_host_float = TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int >;
using Multidiagonal_host_int = TNL::Matrices::MultidiagonalMatrix< int, TNL::Devices::Host, int >;

using Multidiagonal_cuda_float = TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int >;
using Multidiagonal_cuda_int = TNL::Matrices::MultidiagonalMatrix< int, TNL::Devices::Cuda, int >;

static const char* TEST_FILE_NAME = "test_MultidiagonalMatrixTest.tnl";

#ifdef HAVE_GTEST
#include <type_traits>

#include <gtest/gtest.h>

void test_GetSerializationType()
{
   using namespace TNL::Containers::Segments;
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< float, [any_device], int, true, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< int,   TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< int, [any_device], int, true, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< float, [any_device], int, true, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< int,   TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< int, [any_device], int, true, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< float, [any_device], int, false, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< int,   TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< int, [any_device], int, false, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< float, [any_device], int, false, [any_allocator], [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::MultidiagonalMatrix< int,   TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::MultidiagonalMatrix< int, [any_device], int, false, [any_allocator], [any_allocator] >" ) );
}

template< typename Matrix >
void test_SetDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   const IndexType rows = 9;
   const IndexType cols = 8;
   const DiagonalsShiftsType diagonalsShifts{ -3, -1, 0, 2, 4 };

   Matrix m;
   m.setDimensions( rows, cols, diagonalsShifts );

   EXPECT_EQ( m.getRows(), 9 );
   EXPECT_EQ( m.getColumns(), 8 );
}


template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;
   using DiagonalsShiftsType = typename Matrix1::DiagonalsShiftsType;

   const IndexType rows = 8;
   const IndexType cols = 7;
   const DiagonalsShiftsType diagonalsShifts{ -3, -1, 0, 2, 4 };

   Matrix1 m1;
   m1.setDimensions( rows + 1, cols + 2, diagonalsShifts );

   Matrix2 m2;
   m2.setDimensions( rows, cols, diagonalsShifts );

   m1.setLike( m2 );

   EXPECT_EQ( m1.getRows(), m2.getRows() );
   EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void test_SetElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const int gridSize( 4 );
   const int matrixSize( gridSize * gridSize );
   Matrix matrix( matrixSize, matrixSize, { - gridSize, -1, 0, 1, gridSize } );
   matrix.setElements( {
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      { -1.0, -1.0, 4.0, -1.0, -1.0 },
      { -1.0, -1.0, 4.0, -1.0, -1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      { -1.0, -1.0, 4.0, -1.0, -1.0 },
      { -1.0, -1.0, 4.0, -1.0, -1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 },
      {  0.0,  0.0, 1.0 }
   } );

   std::cerr << matrix << std::endl;
   for( int i = 0; i < gridSize; i++ )
      for( int j = 0; j < gridSize; j++ )
      {
         const int elementIdx = i * gridSize + j;
         if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )  // check matrix elements corresponding to boundary grid nodes
         {
            for( int k = 0; k < matrixSize; k++ )
            {
               if( elementIdx == k )
                  EXPECT_EQ( matrix.getElement( elementIdx, k ), 1.0 );
               else
                  EXPECT_EQ( matrix.getElement( elementIdx, k ), 0.0 );
            }
         }
         else // check matrix elements corresponding to inner grid nodes
         {
            for( int k = 0; k < matrixSize; k++ )
            {
               if( k == elementIdx - gridSize || 
                   k == elementIdx - 1 ||
                   k == elementIdx + 1 ||
                   k == elementIdx + gridSize )
                  EXPECT_EQ( matrix.getElement( elementIdx, k ), -1.0 );
               else if( k == elementIdx )
                  EXPECT_EQ( matrix.getElement( elementIdx, k ), 4.0 );
               else
                  EXPECT_EQ( matrix.getElement( elementIdx, k ), 0.0 );
            }
         }
      }
}

template< typename Matrix >
void test_GetNonemptyRowsCount()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x8 matrix:
    *
    *    /  1  0  0  1  0  1  0  0 \
    *    |  0  1  0  0  1  0  1  0 |
    *    |  1  0  1  0  0  1  0  1 |
    *    |  0  1  0  1  0  0  1  0 |
    *    \  0  0  1  0  1  0  0  1 /
    */
   Matrix m1( 5, 8, DiagonalsShiftsType({ -2, 0, 3, 5 }) );
   m1.setValue( 1.0 );
   EXPECT_EQ( m1.getNonemptyRowsCount(), 5 );

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  0  0  1  0  \
    *    |  0  1  0  0  1  |
    *    |  1  0  1  0  0  |
    *    |  0  1  0  1  0  |
    *    \  0  0  1  0  1  /
    */
   Matrix m2( 5, 5, DiagonalsShiftsType({ -2, 0, 3, 5 }) );
   m2.setValue( 1.0 );
   EXPECT_EQ( m2.getNonemptyRowsCount(), 5 );

   /*
    * Sets up the following 8x5 matrix:
    *
    *    /  1  0  0  1  0  \
    *    |  0  1  0  0  1  |
    *    |  1  0  1  0  0  |
    *    |  0  1  0  1  0  |
    *    |  0  0  1  0  1  |
    *    |  0  0  0  1  0  |
    *    |  0  0  0  0  1  |
    *    \  0  0  0  0  0  /
    */
   Matrix m3( 8, 5, DiagonalsShiftsType({ -2, 0, 3, 5 }) );
   m3.setValue( 1.0 );
   EXPECT_EQ( m3.getNonemptyRowsCount(), 7 );
}

template< typename Matrix >
void test_GetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 8x8 matrix:
    *
    *    /  0  0  0  1  0  1  0  0 \  -> 2
    *    |  0  1  0  0  1  0  1  0 |  -> 3
    *    |  1  0  1  0  0  1  0  1 |  -> 4
    *    |  0  1  0  1  0  0  1  0 |  -> 3
    *    |  0  0  1  0  1  0  0  1 |  -> 3
    *    |  0  0  0  1  0  1  0  0 |  -> 2
    *    |  0  0  0  0  1  0  1  0 |  -> 2
    *    \  0  0  0  0  0  1  0  0 /  -> 1
    */

   const IndexType rows = 8;
   const IndexType cols = 8;

   Matrix m( rows, cols, DiagonalsShiftsType({ -2, 0, 3, 5 }) );
   m.setValue( 1.0 );
   m.setElement( 0, 0, 0.0 );
   m.setElement( 7, 7, 0.0 );

   typename Matrix::CompressedRowLengthsVector rowLengths( rows );
   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::CompressedRowLengthsVector correctRowLengths{ 2, 3, 4, 3, 3, 2, 2, 1 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix >
void test_GetAllocatedElementsCount()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m1( 7, 6, DiagonalsShiftsType( { -2, 0, 3, 5 } ) );
   EXPECT_EQ( m1.getAllocatedElementsCount(), 28 );

   Matrix m2( 8, 6, DiagonalsShiftsType( { -2, 0, 3, 5 } ) );
   EXPECT_EQ( m2.getAllocatedElementsCount(), 32 );

   Matrix m3( 9, 6, DiagonalsShiftsType( { -2, 0, 3, 5 } ) );
   EXPECT_EQ( m3.getAllocatedElementsCount(), 32 );
}

template< typename Matrix >
void test_GetNumberOfNonzeroMatrixElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 7x6 matrix:
    *
    *    /  0  0  1  0  1  0 \ -> 2
    *    |  0  1  0  1  0  1 | -> 3
    *    |  0  0  1  0  1  0 | -> 2
    *    |  1  0  0  1  0  1 | -> 3
    *    |  0  1  0  0  1  0 | -> 2
    *    |  0  0  1  0  0  1 | -> 2
    *    \  0  0  0  1  0  0 / -> 1
    *                           ----
    *                            15
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m( rows, cols, DiagonalsShiftsType( { -3, 0, 2, 4 } ) );
   m.setValue( 1.0 );
   m.setElement( 0, 0, 0.0 );

   EXPECT_EQ( m.getNumberOfNonzeroMatrixElements(), 15 );
}

template< typename Matrix >
void test_Reset()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x4 matrix:
    *
    *    /  0  0  0  0 \
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    \  0  0  0  0 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m( rows, cols, DiagonalsShiftsType( { 0, 1, 2, 4 } ) );
   m.reset();

   EXPECT_EQ( m.getRows(), 0 );
   EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_SetValue()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 7x6 matrix:
    *
    *    /  1  0  1  0  1  0 \
    *    |  0  1  0  1  0  1 |
    *    |  0  0  1  0  1  0 |
    *    |  1  0  0  1  0  1 |
    *    |  0  1  0  0  1  0 |
    *    |  0  0  1  0  0  1 |
    *    \  0  0  0  1  0  0 /
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m( rows, cols, DiagonalsShiftsType( { -3, 0, 2, 4 } ) );
   m.setValue( 1.0 );

   EXPECT_EQ( m.getElement( 0, 0 ), 1 );
   EXPECT_EQ( m.getElement( 0, 1 ), 0 );
   EXPECT_EQ( m.getElement( 0, 2 ), 1 );
   EXPECT_EQ( m.getElement( 0, 3 ), 0 );
   EXPECT_EQ( m.getElement( 0, 4 ), 1 );
   EXPECT_EQ( m.getElement( 0, 5 ), 0 );

   EXPECT_EQ( m.getElement( 1, 0 ), 0 );
   EXPECT_EQ( m.getElement( 1, 1 ), 1 );
   EXPECT_EQ( m.getElement( 1, 2 ), 0 );
   EXPECT_EQ( m.getElement( 1, 3 ), 1 );
   EXPECT_EQ( m.getElement( 1, 4 ), 0 );
   EXPECT_EQ( m.getElement( 1, 5 ), 1 );

   EXPECT_EQ( m.getElement( 2, 0 ), 0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 1 );
   EXPECT_EQ( m.getElement( 2, 3 ), 0 );
   EXPECT_EQ( m.getElement( 2, 4 ), 1 );
   EXPECT_EQ( m.getElement( 2, 5 ), 0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 1 );
   EXPECT_EQ( m.getElement( 3, 1 ), 0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 1 );
   EXPECT_EQ( m.getElement( 3, 4 ), 0 );
   EXPECT_EQ( m.getElement( 3, 5 ), 1 );

   EXPECT_EQ( m.getElement( 4, 0 ), 0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 1 );
   EXPECT_EQ( m.getElement( 4, 2 ), 0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 1 );
   EXPECT_EQ( m.getElement( 4, 5 ), 0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 0 );
   EXPECT_EQ( m.getElement( 5, 1 ), 0 );
   EXPECT_EQ( m.getElement( 5, 2 ), 1 );
   EXPECT_EQ( m.getElement( 5, 3 ), 0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 0 );
   EXPECT_EQ( m.getElement( 5, 5 ), 1 );

   EXPECT_EQ( m.getElement( 6, 0 ), 0 );
   EXPECT_EQ( m.getElement( 6, 1 ), 0 );
   EXPECT_EQ( m.getElement( 6, 2 ), 0 );
   EXPECT_EQ( m.getElement( 6, 3 ), 1 );
   EXPECT_EQ( m.getElement( 6, 4 ), 0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 0 );
}

template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  0  5 \
    *    |  0  7  8  0  0 |
    *    |  0  0 13 14  0 |
    *    | 16  0  0 19 20 |
    *    \  0 22  0  0 25 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;
   DiagonalsShiftsType diagonals{-3, 0, 1, 4 };
   Matrix m( rows, cols, diagonals );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         if( diagonals.containsValue( j - i ) )
            m.setElement( i, j, value++ );
         else
         {
            EXPECT_THROW( m.setElement( i, j, value++ ), std::logic_error );
         }

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  5 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 22 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );
}

template< typename Matrix >
void test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x5 matrix:
    *
    *    /  1  2  0  0  5 \
    *    |  0  7  8  0  0 |
    *    |  0  0 13 14  0 |
    *    |  0  0  0 19 20 |
    *    \  0  0  0  0 25 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;
   DiagonalsShiftsType diagonals{-3, 0, 1, 4 };
   Matrix m( rows, cols, diagonals );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         if( diagonals.containsValue( j - i ) )
         {
            if( j >= i )
               m.setElement( i, j, value );
            value++;
         }
         else
         {
            EXPECT_THROW( m.setElement( i, j, value++ ), std::logic_error );
         }

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  5 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 matrix:
    *
    *     /  1  2  0  0  5 \   /  1  2  0  0  5 \    /  3  6  0  0 15 \
    *     |  0  7  8  0  0 |   |  0  7  8  0  0 |    |  0 21 24  0  0 |
    * 2 * |  0  0 13 14  0 | + |  0  0 13 14  0 | =  |  0  0 39 42  0 |
    *     |  0  0  0 19 20 |   | 16  0  0 19 20 |    | 16  0  0 57 60 |
    *     \  0  0  0  0 25 /   \  0 22  0  0 25 /    \  0 22  0  0 75 /
    *
    */

   value = 1;
   RealType multiplicator = 2;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         if( diagonals.containsValue( j - i ) )
            m.addElement( i, j, value++, multiplicator );
         else
         {
            EXPECT_THROW( m.addElement( i, j, value++, multiplicator ), std::logic_error );
         }

   EXPECT_EQ( m.getElement( 0, 0 ),  3 );
   EXPECT_EQ( m.getElement( 0, 1 ),  6 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ), 15 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ), 21 );
   EXPECT_EQ( m.getElement( 1, 2 ), 24 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 39 );
   EXPECT_EQ( m.getElement( 2, 3 ), 42 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 57 );
   EXPECT_EQ( m.getElement( 3, 4 ), 60 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 22 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 75 );
}

template< typename Matrix >
void test_SetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x7 matrix:
    *
    *    /  1  0  2  0  3  0  0 \
    *    |  4  5  0  6  0  7  0 |
    *    |  0  8  9  0 10  0 11 |
    *    |  0  0 12 13  0 14  0 |
    *    \  0  0  0 15 16  0 17 /
    */
   const IndexType rows = 5;
   const IndexType cols = 7;

   Matrix m( rows, cols, DiagonalsShiftsType({ -1, 0, 2, 4 }) );

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 5 ][ 4 ] {
         {  0,  1,  2,  3 },
         {  4,  5,  6,  7 },
         {  8,  9, 10, 11 },
         { 12, 13, 14,  0 },
         { 15, 16, 17,  0 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 4; i++ )
         row.setElement( i, values[ rowIdx ][ i ] );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType) 0, rows, f );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  2 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  3 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  4 );
   EXPECT_EQ( m.getElement( 1, 1 ),  5 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  7 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  8 );
   EXPECT_EQ( m.getElement( 2, 2 ),  9 );
   EXPECT_EQ( m.getElement( 2, 3 ),  0 );
   EXPECT_EQ( m.getElement( 2, 4 ), 10 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );
   EXPECT_EQ( m.getElement( 2, 6 ), 11 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 12 );
   EXPECT_EQ( m.getElement( 3, 3 ), 13 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ), 14 );
   EXPECT_EQ( m.getElement( 3, 6 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 15 );
   EXPECT_EQ( m.getElement( 4, 4 ), 16 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ), 17 );
}

template< typename Matrix >
void test_AddRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 6x5 matrix:
    *
    *    /  1  2  3  0  0 \
    *    |  0  7  8  9  0 |
    *    |  0  0 13 14 15 |
    *    |  0  0  0 19 20 |
    *    |  0  0  0  0 25 |
    *    \  0  0  0  0  0 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;
   DiagonalsShiftsType diagonals( { -2, 0, 1, 2 } );

   Matrix m( rows, cols, diagonals );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
      {
         IndexType offset = j - i;
         if( diagonals.containsValue( offset ) && offset >= 0)
            m.setElement( i, j, value );
         value++;
      }

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  9 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ), 15 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *  / 0  0  0  0  0  0 \   /  1  2  3  0  0 \   / 11  0  0  0  0 \   / 11   0  0   0   0 \
    *  | 0  1  0  0  0  0 |   |  0  7  8  9  0 |   |  0 22  0  0  0 |   |  0  29  8   9   0 |
    *  | 0  0  2  0  0  0 | * |  0  0 13 14 15 | + | 33  0 33  0  0 | = | 33   0 59  28  30 |
    *  | 0  0  0  3  0  0 |   |  0  0  0 19 20 |   |  0 44  0 44  0 |   |  0  44  0 101  60 |
    *  | 0  0  0  0  4  0 |   |  0  0  0  0 25 |   |  0  0 55  0 55 |   |  0   0 55   0 155 |
    *  \ 0  0  0  0  0  5 /   \  0  0  0  0  0 /   \  0  0  0 66  0 /   \  0   0  0  66   0 /
    */

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 6 ][ 4 ] {
         {  0, 11, 0,  0 },
         {  0, 22, 0,  0 },
         { 33, 33, 0,  0 },
         { 44, 44, 0,  0 },
         { 55, 55, 0,  0 },
         { 66,  0, 0,  0 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 4; i++ )
      {
         RealType& val = row.getValue( i );
         val = rowIdx * val + values[ rowIdx ][ i ];
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( 0, 6, f );

   EXPECT_EQ( m.getElement( 0, 0 ),  11 );
   EXPECT_EQ( m.getElement( 0, 1 ),   0 );
   EXPECT_EQ( m.getElement( 0, 2 ),   0 );
   EXPECT_EQ( m.getElement( 0, 3 ),   0 );
   EXPECT_EQ( m.getElement( 0, 4 ),   0 );

   EXPECT_EQ( m.getElement( 1, 0 ),   0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  29 );
   EXPECT_EQ( m.getElement( 1, 2 ),   8 );
   EXPECT_EQ( m.getElement( 1, 3 ),   9 );
   EXPECT_EQ( m.getElement( 1, 4 ),   0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  33 );
   EXPECT_EQ( m.getElement( 2, 1 ),   0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  59 );
   EXPECT_EQ( m.getElement( 2, 3 ),  28 );
   EXPECT_EQ( m.getElement( 2, 4 ),  30  );

   EXPECT_EQ( m.getElement( 3, 0 ),   0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  44 );
   EXPECT_EQ( m.getElement( 3, 2 ),   0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 101 );
   EXPECT_EQ( m.getElement( 3, 4 ),  60 );

   EXPECT_EQ( m.getElement( 4, 0 ),   0 );
   EXPECT_EQ( m.getElement( 4, 1 ),   0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  55 );
   EXPECT_EQ( m.getElement( 4, 3 ),   0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 155 );

   EXPECT_EQ( m.getElement( 5, 0 ),   0 );
   EXPECT_EQ( m.getElement( 5, 1 ),   0 );
   EXPECT_EQ( m.getElement( 5, 2 ),   0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  66 );
   EXPECT_EQ( m.getElement( 5, 4 ),   0 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x4 matrix:
    *
    *    /  1  0  3  0 \
    *    |  0  6  0  8 |
    *    |  9  0 11  0 |
    *    |  0 14  0 16 |
    *    \  0  0 19  0 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;
   DiagonalsShiftsType diagonals{ -2, 0, 2 };

   Matrix m( rows, cols, diagonals );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
      {
         if( diagonals.containsValue( j - i ) )
            m.setElement( i, j, value );
         value++;
      }

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   VectorType inVector( 4 );
   inVector = 2;

   VectorType outVector( 5 );
   outVector = 0;

   m.vectorProduct( inVector, outVector);

   EXPECT_EQ( outVector.getElement( 0 ),  8 );
   EXPECT_EQ( outVector.getElement( 1 ), 28 );
   EXPECT_EQ( outVector.getElement( 2 ), 40 );
   EXPECT_EQ( outVector.getElement( 3 ), 60 );
   EXPECT_EQ( outVector.getElement( 4 ), 38 );
}

template< typename Matrix1, typename Matrix2 = Matrix1 >
void test_AddMatrix()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;
   using DiagonalsShiftsType1 = typename Matrix1::DiagonalsShiftsType;
   using DiagonalsShiftsType2 = typename Matrix2::DiagonalsShiftsType;

   /*
    * Sets up the following 5x4 matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    |  0  0 15 16 |
    *    \  0  0  0 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;
   DiagonalsShiftsType1 diagonals1;
   DiagonalsShiftsType2 diagonals2;

   Matrix1 m( rows, cols, diagonals1 );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
      {
         if( abs( i - j ) <= 1 )
            m.setElement( i, j, value );
         value++;
      }

   /*
    * Sets up the following 5x4 matrix:
    *
    *    /  1  2  0  0 \
    *    |  3  4  5  0 |
    *    |  0  6  7  8 |
    *    |  0  0  9 10 |
    *    \  0  0  0 11 /
    */
   Matrix2 m2( rows, cols, diagonals2 );

   RealType newValue = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
         if( abs( i - j ) <= 1 )
            m2.setElement( i, j, newValue++ );

   /*
    * Compute the following 5x4 matrix:
    *
    *  /  1  2  0  0 \       /  1  2  0  0 \    /  3  6  0  0 \
    *  |  5  6  7  0 |       |  3  4  5  0 |    | 11 14 17  0 |
    *  |  0 10 11 12 | + 2 * |  0  6  7  8 | =  |  0 22 25 28 |
    *  |  0  0 15 16 |       |  0  0  9 10 |    |  0  0 33 36 |
    *  \  0  0  0 20 /       \  0  0  0 11 /    \  0  0  0 42 /
    */

   Matrix1 mResult;
   mResult.reset();
   mResult.setDimensions( rows, cols );

   mResult = m;

   RealType matrixMultiplicator = 2;
   RealType thisMatrixMultiplicator = 1;

   mResult.addMatrix( m2, matrixMultiplicator, thisMatrixMultiplicator );

   EXPECT_EQ( mResult.getElement( 0, 0 ), matrixMultiplicator * m2.getElement( 0, 0 ) + thisMatrixMultiplicator * m.getElement( 0, 0 ) );
   EXPECT_EQ( mResult.getElement( 0, 1 ), matrixMultiplicator * m2.getElement( 0, 1 ) + thisMatrixMultiplicator * m.getElement( 0, 1 ) );
   EXPECT_EQ( mResult.getElement( 0, 2 ), matrixMultiplicator * m2.getElement( 0, 2 ) + thisMatrixMultiplicator * m.getElement( 0, 2 ) );
   EXPECT_EQ( mResult.getElement( 0, 3 ), matrixMultiplicator * m2.getElement( 0, 3 ) + thisMatrixMultiplicator * m.getElement( 0, 3 ) );

   EXPECT_EQ( mResult.getElement( 1, 0 ), matrixMultiplicator * m2.getElement( 1, 0 ) + thisMatrixMultiplicator * m.getElement( 1, 0 ) );
   EXPECT_EQ( mResult.getElement( 1, 1 ), matrixMultiplicator * m2.getElement( 1, 1 ) + thisMatrixMultiplicator * m.getElement( 1, 1 ) );
   EXPECT_EQ( mResult.getElement( 1, 2 ), matrixMultiplicator * m2.getElement( 1, 2 ) + thisMatrixMultiplicator * m.getElement( 1, 2 ) );
   EXPECT_EQ( mResult.getElement( 1, 3 ), matrixMultiplicator * m2.getElement( 1, 3 ) + thisMatrixMultiplicator * m.getElement( 1, 3 ) );

   EXPECT_EQ( mResult.getElement( 2, 0 ), matrixMultiplicator * m2.getElement( 2, 0 ) + thisMatrixMultiplicator * m.getElement( 2, 0 ) );
   EXPECT_EQ( mResult.getElement( 2, 1 ), matrixMultiplicator * m2.getElement( 2, 1 ) + thisMatrixMultiplicator * m.getElement( 2, 1 ) );
   EXPECT_EQ( mResult.getElement( 2, 2 ), matrixMultiplicator * m2.getElement( 2, 2 ) + thisMatrixMultiplicator * m.getElement( 2, 2 ) );
   EXPECT_EQ( mResult.getElement( 2, 3 ), matrixMultiplicator * m2.getElement( 2, 3 ) + thisMatrixMultiplicator * m.getElement( 2, 3 ) );

   EXPECT_EQ( mResult.getElement( 3, 0 ), matrixMultiplicator * m2.getElement( 3, 0 ) + thisMatrixMultiplicator * m.getElement( 3, 0 ) );
   EXPECT_EQ( mResult.getElement( 3, 1 ), matrixMultiplicator * m2.getElement( 3, 1 ) + thisMatrixMultiplicator * m.getElement( 3, 1 ) );
   EXPECT_EQ( mResult.getElement( 3, 2 ), matrixMultiplicator * m2.getElement( 3, 2 ) + thisMatrixMultiplicator * m.getElement( 3, 2 ) );
   EXPECT_EQ( mResult.getElement( 3, 3 ), matrixMultiplicator * m2.getElement( 3, 3 ) + thisMatrixMultiplicator * m.getElement( 3, 3 ) );

   EXPECT_EQ( mResult.getElement( 4, 0 ), matrixMultiplicator * m2.getElement( 4, 0 ) + thisMatrixMultiplicator * m.getElement( 4, 0 ) );
   EXPECT_EQ( mResult.getElement( 4, 1 ), matrixMultiplicator * m2.getElement( 4, 1 ) + thisMatrixMultiplicator * m.getElement( 4, 1 ) );
   EXPECT_EQ( mResult.getElement( 4, 2 ), matrixMultiplicator * m2.getElement( 4, 2 ) + thisMatrixMultiplicator * m.getElement( 4, 2 ) );
   EXPECT_EQ( mResult.getElement( 4, 3 ), matrixMultiplicator * m2.getElement( 4, 3 ) + thisMatrixMultiplicator * m.getElement( 4, 3 ) );

   EXPECT_EQ( mResult.getElement( 0, 0 ),  3 );
   EXPECT_EQ( mResult.getElement( 0, 1 ),  6 );
   EXPECT_EQ( mResult.getElement( 0, 2 ),  0 );
   EXPECT_EQ( mResult.getElement( 0, 3 ),  0 );

   EXPECT_EQ( mResult.getElement( 1, 0 ), 11 );
   EXPECT_EQ( mResult.getElement( 1, 1 ), 14 );
   EXPECT_EQ( mResult.getElement( 1, 2 ), 17 );
   EXPECT_EQ( mResult.getElement( 1, 3 ),  0 );

   EXPECT_EQ( mResult.getElement( 2, 0 ),  0 );
   EXPECT_EQ( mResult.getElement( 2, 1 ), 22 );
   EXPECT_EQ( mResult.getElement( 2, 2 ), 25 );
   EXPECT_EQ( mResult.getElement( 2, 3 ), 28 );

   EXPECT_EQ( mResult.getElement( 3, 0 ),  0 );
   EXPECT_EQ( mResult.getElement( 3, 1 ),  0 );
   EXPECT_EQ( mResult.getElement( 3, 2 ), 33 );
   EXPECT_EQ( mResult.getElement( 3, 3 ), 36 );

   EXPECT_EQ( mResult.getElement( 4, 0 ),  0 );
   EXPECT_EQ( mResult.getElement( 4, 1 ),  0 );
   EXPECT_EQ( mResult.getElement( 4, 2 ),  0 );
   EXPECT_EQ( mResult.getElement( 4, 3 ), 42 );
}

template< typename Matrix >
void test_GetMatrixProduct()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
    using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;
/*
 * Sets up the following 5x4 matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const IndexType leftRows = 5;
    const IndexType leftCols = 4;
    DiagonalsShiftsType diagonalsShifts( { 0, 1, 2 } );

    Matrix leftMatrix( leftRows, leftCols, diagonalsShifts );

    RealType value = 1;
    for( IndexType i = 0; i < leftRows; i++ )
        for( IndexType j = 0; j < leftCols; j++)
            leftMatrix.setElement( i, j, value++ );

/*
 * Sets up the following 4x5 matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    \ 16 17 18 19 20 /
 */
    const IndexType rightRows = 4;
    const IndexType rightCols = 5;

    Matrix rightMatrix;
    rightMatrix.reset();
    rightMatrix.setDimensions( rightRows, rightCols );

    RealType newValue = 1;
    for( IndexType i = 0; i < rightRows; i++ )
        for( IndexType j = 0; j < rightCols; j++)
            rightMatrix.setElement( i, j, newValue++ );

/*
 * Sets up the following 5x5 resulting matrix:
 *
 *    /  0  0  0  0 \
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    \  0  0  0  0 /
 */

    Matrix mResult( leftRows, rightCols, diagonalsShifts );
    mResult.setValue( 0 );

    RealType leftMatrixMultiplicator = 1;
    RealType rightMatrixMultiplicator = 2;
/*
 *      /  1  2  3  4 \                            /  220  240  260  280  300 \
 *      |  5  6  7  8 |       /  1  2  3  4  5 \   |  492  544  596  648  700 |
 *  1 * |  9 10 11 12 | * 2 * |  6  7  8  9 10 | = |  764  848  932 1016 1100 |
 *      | 13 14 15 16 |       | 11 12 13 14 15 |   | 1036 1152 1268 1384 1500 |
 *      \ 17 18 19 20 /       \ 16 17 18 19 20 /   \ 1308 1456 1604 1752 1900 /
 */

    mResult.getMatrixProduct( leftMatrix, rightMatrix, leftMatrixMultiplicator, rightMatrixMultiplicator );

    EXPECT_EQ( mResult.getElement( 0, 0 ),  220 );
    EXPECT_EQ( mResult.getElement( 0, 1 ),  240 );
    EXPECT_EQ( mResult.getElement( 0, 2 ),  260 );
    EXPECT_EQ( mResult.getElement( 0, 3 ),  280 );
    EXPECT_EQ( mResult.getElement( 0, 4 ),  300 );

    EXPECT_EQ( mResult.getElement( 1, 0 ),  492 );
    EXPECT_EQ( mResult.getElement( 1, 1 ),  544 );
    EXPECT_EQ( mResult.getElement( 1, 2 ),  596 );
    EXPECT_EQ( mResult.getElement( 1, 3 ),  648 );
    EXPECT_EQ( mResult.getElement( 1, 4 ),  700 );

    EXPECT_EQ( mResult.getElement( 2, 0 ),  764 );
    EXPECT_EQ( mResult.getElement( 2, 1 ),  848 );
    EXPECT_EQ( mResult.getElement( 2, 2 ),  932 );
    EXPECT_EQ( mResult.getElement( 2, 3 ), 1016 );
    EXPECT_EQ( mResult.getElement( 2, 4 ), 1100 );

    EXPECT_EQ( mResult.getElement( 3, 0 ), 1036 );
    EXPECT_EQ( mResult.getElement( 3, 1 ), 1152 );
    EXPECT_EQ( mResult.getElement( 3, 2 ), 1268 );
    EXPECT_EQ( mResult.getElement( 3, 3 ), 1384 );
    EXPECT_EQ( mResult.getElement( 3, 4 ), 1500 );

    EXPECT_EQ( mResult.getElement( 4, 0 ), 1308 );
    EXPECT_EQ( mResult.getElement( 4, 1 ), 1456 );
    EXPECT_EQ( mResult.getElement( 4, 2 ), 1604 );
    EXPECT_EQ( mResult.getElement( 4, 3 ), 1752 );
    EXPECT_EQ( mResult.getElement( 4, 4 ), 1900 );
}

template< typename Matrix >
void test_GetTransposition()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
    using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;
/*
 * Sets up the following 3x2 matrix:
 *
 *    /  1  2 \
 *    |  3  4 |
 *    \  5  6 /
 */
    const IndexType rows = 3;
    const IndexType cols = 2;
    DiagonalsShiftsType diagonalsShifts( { 0, 1, 2 } );

    Matrix m( rows, cols, diagonalsShifts );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );

    m.print( std::cout );

/*
 * Sets up the following 2x3 matrix:
 *
 *    /  0  0  0 \
 *    \  0  0  0 /
 */
    Matrix mTransposed( cols, rows, diagonalsShifts );

    mTransposed.print( std::cout );

    RealType matrixMultiplicator = 1;

    mTransposed.getTransposition( m, matrixMultiplicator );

    mTransposed.print( std::cout );

/*
 * Should result in the following 2x3 matrix:
 *
 *    /  1  3  5 \
 *    \  2  4  6 /
 */

    EXPECT_EQ( mTransposed.getElement( 0, 0 ), 1 );
    EXPECT_EQ( mTransposed.getElement( 0, 1 ), 3 );
    EXPECT_EQ( mTransposed.getElement( 0, 2 ), 5 );

    EXPECT_EQ( mTransposed.getElement( 1, 0 ), 2 );
    EXPECT_EQ( mTransposed.getElement( 1, 1 ), 4 );
    EXPECT_EQ( mTransposed.getElement( 1, 2 ), 6 );
}


template< typename Matrix >
void test_PerformSORIteration()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
    using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;
/*
 * Sets up the following 4x4 matrix:
 *
 *    /  4  1  1  1 \
 *    |  1  4  1  1 |
 *    |  1  1  4  1 |
 *    \  1  1  1  4 /
 */
    const IndexType rows = 4;
    const IndexType cols = 4;
    DiagonalsShiftsType diagonalsShifts( { 0, 1, 2 } );

    Matrix m( rows, cols, diagonalsShifts );

    m.setElement( 0, 0, 4.0 );        // 0th row
    m.setElement( 0, 1, 1.0 );
    m.setElement( 0, 2, 1.0 );
    m.setElement( 0, 3, 1.0 );

    m.setElement( 1, 0, 1.0 );        // 1st row
    m.setElement( 1, 1, 4.0 );
    m.setElement( 1, 2, 1.0 );
    m.setElement( 1, 3, 1.0 );

    m.setElement( 2, 0, 1.0 );
    m.setElement( 2, 1, 1.0 );        // 2nd row
    m.setElement( 2, 2, 4.0 );
    m.setElement( 2, 3, 1.0 );

    m.setElement( 3, 0, 1.0 );        // 3rd row
    m.setElement( 3, 1, 1.0 );
    m.setElement( 3, 2, 1.0 );
    m.setElement( 3, 3, 4.0 );

    RealType bVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };
    RealType xVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };

    IndexType row = 0;
    RealType omega = 1;

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ],  1.0 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  0.15625 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ], 0.15625 );
    EXPECT_EQ( xVector[ 3 ], 0.3671875 );
}

template< typename Matrix >
void test_AssignmentOperator()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;
   constexpr TNL::Containers::Segments::ElementsOrganization organization = Matrix::getOrganization();

   using MultidiagonalHost = TNL::Matrices::MultidiagonalMatrix< RealType, TNL::Devices::Host, IndexType, organization >;
   using MultidiagonalCuda = TNL::Matrices::MultidiagonalMatrix< RealType, TNL::Devices::Cuda, IndexType,
      organization == TNL::Containers::Segments::RowMajorOrder ? TNL::Containers::Segments::ColumnMajorOrder : TNL::Containers::Segments::RowMajorOrder >;

   const IndexType rows( 10 ), columns( 10 );
   DiagonalsShiftsType diagonalsShifts( { -4, -2, 0, 2, 3, 5 } );
   MultidiagonalHost hostMatrix( rows, columns, diagonalsShifts );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j <  columns; j++ )
         if( diagonalsShifts.containsValue( j - i ) )
            hostMatrix.setElement( i, j,  i + j );

   Matrix matrix( rows, columns, diagonalsShifts );
   matrix.getValues() = 0.0;
   matrix = hostMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ )
            if( diagonalsShifts.containsValue( j - i ) )
               EXPECT_EQ( matrix.getElement( i, j ), i + j );
            else
               EXPECT_EQ( matrix.getElement( i, j ), 0.0 );

#ifdef HAVE_CUDA
   MultidiagonalCuda cudaMatrix( rows, columns, diagonalsShifts );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
         if( diagonalsShifts.containsValue( j - i ) )
            cudaMatrix.setElement( i, j, i + j );

   matrix.getValues() = 0.0;
   matrix = cudaMatrix;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( diagonalsShifts.containsValue( j - i ) )
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
         else
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
      }
#endif
}


template< typename Matrix >
void test_SaveAndLoad()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 4x4 matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    \  0  0 15 16 /
    */
   const IndexType rows = 4;
   const IndexType cols = 4;
   DiagonalsShiftsType diagonalsShifts( { -1, 0, 1 } );

   Matrix savedMatrix( rows, cols, diagonalsShifts );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
      {
         if( diagonalsShifts.containsValue( j - i ) )
            savedMatrix.setElement( i, j, value );
         value++;
      }

   ASSERT_NO_THROW( savedMatrix.save( TEST_FILE_NAME ) );

   Matrix loadedMatrix;

   ASSERT_NO_THROW( loadedMatrix.load( TEST_FILE_NAME ) );

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
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  5 );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  6 );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  7 );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), 10 );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), 11 );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), 12 );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), 15 );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), 16 );
}

template< typename Matrix >
void test_Print()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DiagonalsShiftsType = typename Matrix::DiagonalsShiftsType;

   /*
    * Sets up the following 5x4 sparse matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    |  0  0 15 16 |
    *    \  0  0  0 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;
   DiagonalsShiftsType diagonalsShifts( { -1, 0, 1 } );

   Matrix m( rows, cols, diagonalsShifts );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++)
      for( IndexType j = 0; j < cols; j++)
      {
         if( abs( i - j ) <= 1 )
            m.setElement( i, j, value );
         value++;
      }

   std::stringstream printed;
   std::stringstream couted;

   //change the underlying buffer and save the old buffer
   auto old_buf = std::cout.rdbuf(printed.rdbuf());

   m.print( std::cout ); //all the std::cout goes to ss

   std::cout.rdbuf(old_buf); //reset
   couted << "Row: 0 ->  Col:0->1\t Col:1->2\t\n"
             "Row: 1 ->  Col:0->5\t Col:1->6\t Col:2->7\t\n"
             "Row: 2 ->  Col:1->10\t Col:2->11\t Col:3->12\t\n"
             "Row: 3 ->  Col:2->15\t Col:3->16\t\n"
             "Row: 4 ->  Col:3->20\t\n";

   EXPECT_EQ( printed.str(), couted.str() );
}

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types
<
    TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Host, short >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Host, short >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Host, short >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, short >,
    TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Host, int >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Host, int >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Host, int >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int >,
    TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Host, long >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Host, long >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Host, long >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
    ,TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::MultidiagonalMatrix< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::MultidiagonalMatrix< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::MultidiagonalMatrix< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, getSerializationType )
{
   test_GetSerializationType();
}

TYPED_TEST( MatrixTest, setDimensionsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetDimensions< MatrixType >();
}

TYPED_TEST( MatrixTest, setLikeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST( MatrixTest, setElements )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetElements< MatrixType >();
}

TYPED_TEST( MatrixTest, getNonemptyRowsCountTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetNonemptyRowsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, getCompressedRowLengthTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetCompressedRowLengths< MatrixType >();
}

TYPED_TEST( MatrixTest, getAllocatedElementsCountTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetAllocatedElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, getNumberOfNonzeroMatrixElementsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetNumberOfNonzeroMatrixElements< MatrixType >();
}

TYPED_TEST( MatrixTest, resetTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Reset< MatrixType >();
}

TYPED_TEST( MatrixTest, setValueTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetValue< MatrixType >();
}

TYPED_TEST( MatrixTest, setElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetElement< MatrixType >();
}

TYPED_TEST( MatrixTest, addElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddElement< MatrixType >();
}

TYPED_TEST( MatrixTest, setRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetRow< MatrixType >();
}

TYPED_TEST( MatrixTest, addRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddRow< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct< MatrixType >();
}

/*TYPED_TEST( MatrixTest, addMatrixTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddMatrix< MatrixType >();
}*/

TYPED_TEST( MatrixTest, assignmentOperatorTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AssignmentOperator< MatrixType >();
}

TYPED_TEST( MatrixTest, saveAndLoadTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SaveAndLoad< MatrixType >();
}

TYPED_TEST( MatrixTest, printTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Print< MatrixType >();
}

/*TEST( MultidiagonalMatrixTest, Multidiagonal_getMatrixProductTest_Host )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h(609): error: no instance of function template \"TNL::Matrices::MultidiagonalMatrixProductKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Multidiagonal<int, TNL::Devices::Host, int> *, Multidiagonal_host_int *, Multidiagonal_host_int *, const int, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Multidiagonal<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix1=Multidiagonal_host_int, Matrix2=Multidiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Multidiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(1315): here\n\n";
}

#ifdef HAVE_CUDA
TEST( MultidiagonalMatrixTest, Multidiagonal_getMatrixProductTest_Cuda )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h(510): error: identifier \"tnlCudaMin\" is undefined\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::MultidiagonalMatrixProductKernel<Real,Index,Matrix1,Matrix2,tileDim,tileRowBlockSize>(TNL::Matrices::Multidiagonal<Real, TNL::Devices::Cuda, Index> *, const Matrix1 *, const Matrix2 *, Real, Real, Index, Index) [with Real=int, Index=int, Matrix1=Multidiagonal_cuda_int, Matrix2=Multidiagonal_cuda_int, tileDim=32, tileRowBlockSize=8]\"\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Multidiagonal<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Cuda, Index=int, Matrix1=Multidiagonal_cuda_int, Matrix2=Multidiagonal_cuda_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Multidiagonal_cuda_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(1332): here\n\n";
}
#endif

TEST( MultidiagonalMatrixTest, Multidiagonal_getTranspositionTest_Host )
{
//    test_GetTransposition< Multidiagonal_host_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h(836): error: no instance of function template \"TNL::Matrices::MultidiagonalTranspositionAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Multidiagonal<int, TNL::Devices::Host, int> *, Multidiagonal_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Multidiagonal<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Multidiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Multidiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(1420): here\n\n";
    std::cout << "AND this message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h(852): error: no instance of function template \"TNL::Matrices::MultidiagonalTranspositionNonAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Multidiagonal<int, TNL::Devices::Host, int> *, Multidiagonal_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Multidiagonal<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Multidiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Multidiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Multidiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/MultidiagonalMatrixTest.h(1420): here\n\n";
}

#ifdef HAVE_CUDA
TEST( MultidiagonalMatrixTest, Multidiagonal_getTranspositionTest_Cuda )
{
//    test_GetTransposition< Multidiagonal_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test throws the following message: \n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!!\n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Multidiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  terminate called after throwing an instance of 'TNL::Exceptions::CudaRuntimeError'\n";
    std::cout << "          what():  CUDA ERROR 4 (cudaErrorLaunchFailure): unspecified launch failure.\n";
    std::cout << "  Source: line 57 in /home/lukas/tnl-dev/src/TNL/Containers/Algorithms/ArrayOperationsCuda_impl.h: unspecified launch failure\n";
    std::cout << "  [1]    4003 abort (core dumped)  ./MultidiagonalMatrixTest-dbg\n";
}
#endif

TEST( MultidiagonalMatrixTest, Multidiagonal_performSORIterationTest_Host )
{
    test_PerformSORIteration< Multidiagonal_host_float >();
}

#ifdef HAVE_CUDA
TEST( MultidiagonalMatrixTest, Multidiagonal_performSORIterationTest_Cuda )
{
//    test_PerformSORIteration< Multidiagonal_cuda_float >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      [1]    6992 segmentation fault (core dumped)  ./SparseMatrixTest-dbg\n\n";
    std::cout << "\n THIS IS NOT IMPLEMENTED FOR CUDA YET!!\n\n";
}
#endif
 * */

#endif // HAVE_GTEST

#include "../main.h"
