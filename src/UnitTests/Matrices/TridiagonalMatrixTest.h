/***************************************************************************
                          TridiagonalMatrixTest.h -  description
                             -------------------
    begin                : Jan 8, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <sstream>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/Tridiagonal.h>
#include <TNL/Containers/Array.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

using Tridiagonal_host_float = TNL::Matrices::Tridiagonal< float, TNL::Devices::Host, int >;
using Tridiagonal_host_int = TNL::Matrices::Tridiagonal< int, TNL::Devices::Host, int >;

using Tridiagonal_cuda_float = TNL::Matrices::Tridiagonal< float, TNL::Devices::Cuda, int >;
using Tridiagonal_cuda_int = TNL::Matrices::Tridiagonal< int, TNL::Devices::Cuda, int >;

static const char* TEST_FILE_NAME = "test_TridiagonalMatrixTest.tnl";

#ifdef HAVE_GTEST
#include <type_traits>

#include <gtest/gtest.h>

void test_GetSerializationType()
{
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< float, TNL::Devices::Host, int, true >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< float, [any_device], int, true, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< int,   TNL::Devices::Host, int, true >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< int, [any_device], int, true, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< float, TNL::Devices::Cuda, int, true >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< float, [any_device], int, true, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< int,   TNL::Devices::Cuda, int, true >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< int, [any_device], int, true, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< float, TNL::Devices::Host, int, false >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< float, [any_device], int, false, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< int,   TNL::Devices::Host, int, false >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< int, [any_device], int, false, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< float, TNL::Devices::Cuda, int, false >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< float, [any_device], int, false, [any_allocator] >" ) );
   EXPECT_EQ( ( TNL::Matrices::Tridiagonal< int,   TNL::Devices::Cuda, int, false >::getSerializationType() ), TNL::String( "Matrices::Tridiagonal< int, [any_device], int, false, [any_allocator] >" ) );
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

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

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
void test_GetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 10;
   const IndexType cols = 11;

   Matrix m( rows, cols );

   // Insert values into the rows.
   RealType value = 1;

   for( IndexType i = 0; i < 2; i++ )  // 0th row -> 2 elements
      m.setElement( 0, i, value++ );

   for( IndexType i = 0; i < 3; i++ )  // 1st row -> 3 elements
      m.setElement( 1, i, value++ );

   for( IndexType i = 1; i < 3; i++ )  // 2nd row -> 2 elements
      m.setElement( 2, i, value++ );

   for( IndexType i = 2; i < 5; i++ )  // 3rd row -> 3 elements
      m.setElement( 3, i, value++ );

   for( IndexType i = 3; i < 6; i++ )  // 4th row -> 3 elements
      m.setElement( 4, i, value++ );

   for( IndexType i = 4; i < 6; i++ )  // 5th row -> 2 elements
      m.setElement( 5, i, value++ );

   for( IndexType i = 5; i < 8; i++ )  // 6th row -> 3 elements
      m.setElement( 6, i, value++ );

   for( IndexType i = 6; i < 8; i++ )  // 7th row -> 2 elements
      m.setElement( 7, i, value++ );

   for( IndexType i = 7; i < 10; i++ ) // 8th row -> 3 elements
      m.setElement( 8, i, value++ );

   for( IndexType i = 8; i < 11; i++ ) // 9th row -> 3 elements
      m.setElement( 9, i, value++ );

   typename Matrix::CompressedRowLengthsVector rowLengths( rows );
   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::CompressedRowLengthsVector correctRowLengths{ 2, 3, 2, 3, 3, 2, 3, 2, 3, 3 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix >
void test_GetRowLength()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

   Matrix m( rows, cols );

   EXPECT_EQ( m.getRowLength( 0 ), 2 );
   EXPECT_EQ( m.getRowLength( 1 ), 3 );
   EXPECT_EQ( m.getRowLength( 2 ), 3 );
   EXPECT_EQ( m.getRowLength( 3 ), 3 );
   EXPECT_EQ( m.getRowLength( 4 ), 3 );
   EXPECT_EQ( m.getRowLength( 5 ), 3 );
   EXPECT_EQ( m.getRowLength( 6 ), 2 );
   EXPECT_EQ( m.getRowLength( 7 ), 1 );
}

template< typename Matrix >
void test_GetAllocatedElementsCount()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m( rows, cols );

   EXPECT_EQ( m.getAllocatedElementsCount(), 21 );
}

template< typename Matrix >
void test_GetNumberOfNonzeroMatrixElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 7x6 dense matrix:
    *
    *    /  0  1  0  0  0  0 \
    *    |  2  3  4  0  0  0 |
    *    |  0  5  6  7  0  0 |
    *    |  0  0  8  9 10  0 |
    *    |  0  0  0 11 12 13 |
    *    |  0  0  0  0 14  0 |
    *    \  0  0  0  0  0 16 /
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m( rows, cols );

   RealType value = 0;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = TNL::max( 0, i - 1 ); j < TNL::min( cols, i + 2 ); j++ )
         m.setElement( i, j, value++ );

   m.setElement( 5, 5, 0);

   EXPECT_EQ( m.getNumberOfNonzeroMatrixElements(), 15 );
}

template< typename Matrix >
void test_Reset()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 dense matrix:
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
void test_SetValue()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 7x6 dense matrix:
    *
    *    /  0  1  0  0  0  0 \
    *    |  2  3  4  0  0  0 |
    *    |  0  5  6  7  0  0 |
    *    |  0  0  8  9 10  0 |
    *    |  0  0  0 11 12 13 |
    *    |  0  0  0  0 14  0 |
    *    \  0  0  0  0  0 16 /
    */
   const IndexType rows = 7;
   const IndexType cols = 6;

   Matrix m( rows, cols );

   RealType value = 0;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = TNL::max( 0, i - 1 ); j < TNL::min( cols, i + 2 ); j++ )
         m.setElement( i, j, value++ );

   m.setElement( 5, 5, 0);

   EXPECT_EQ( m.getElement( 0, 0 ),  0 );
   EXPECT_EQ( m.getElement( 0, 1 ),  1 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  2 );
   EXPECT_EQ( m.getElement( 1, 1 ),  3 );
   EXPECT_EQ( m.getElement( 1, 2 ),  4 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  5 );
   EXPECT_EQ( m.getElement( 2, 2 ),  6 );
   EXPECT_EQ( m.getElement( 2, 3 ),  7 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  8 );
   EXPECT_EQ( m.getElement( 3, 3 ),  9 );
   EXPECT_EQ( m.getElement( 3, 4 ), 10 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 11 );
   EXPECT_EQ( m.getElement( 4, 4 ), 12 );
   EXPECT_EQ( m.getElement( 4, 5 ), 13 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 14 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ),  0 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 16 );

   // Set the values of all elements to a certain number
   m.setValue( 42 );

   EXPECT_EQ( m.getElement( 0, 0 ), 42 );
   EXPECT_EQ( m.getElement( 0, 1 ), 42 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ), 42 );
   EXPECT_EQ( m.getElement( 1, 1 ), 42 );
   EXPECT_EQ( m.getElement( 1, 2 ), 42 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 42 );
   EXPECT_EQ( m.getElement( 2, 2 ), 42 );
   EXPECT_EQ( m.getElement( 2, 3 ), 42 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 42 );
   EXPECT_EQ( m.getElement( 3, 3 ), 42 );
   EXPECT_EQ( m.getElement( 3, 4 ), 42 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 42 );
   EXPECT_EQ( m.getElement( 4, 4 ), 42 );
   EXPECT_EQ( m.getElement( 4, 5 ), 42 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 42 );
   EXPECT_EQ( m.getElement( 5, 5 ), 42 );

   EXPECT_EQ( m.getElement( 6, 0 ),  0 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 42 );
}

template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    /  1  2  0  0  0 \
    *    |  6  7  8  0  0 |
    *    |  0 12 13 14  0 |
    *    |  0  0 18 19 20 |
    *    \  0  0  0 24 25 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
      {
         if( abs( i - j ) > 1 )
         {
            EXPECT_THROW( m.setElement( i, j, value++ ), std::logic_error );
         }
         else
            m.setElement( i, j, value++ );
      }

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );
}

template< typename Matrix >
void test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  0  0  0 \
    *    |  6  7  8  0  0 |
    *    |  0 12 13 14  0 |
    *    |  0  0 18 19 20 |
    *    |  0  0  0 24 25 |
    *    \  0  0  0  0 30 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
        {
           if( abs( i - j ) <= 1 )
               m.setElement( i, j, value );
           value++;
        }

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 30 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 dense matrix:
    *
    *     /  1  2  0  0  0 \    /  1  2  0  0  0 \   /  3  6  0  0  0 \
    *     |  6  7  8  0  0 |    |  3  4  5  0  0 |   | 15 18 21  0  0 |
    * 2 * |  0 12 13 14  0 |  + |  0  6  7  8  0 | = |  0 30 33 36  0 |
    *     |  0  0 18 19 20 |    |  0  0  9 10 11 |   |  0  0 45 48 51 |
    *     |  0  0  0 24 25 |    |  0  0  0 12 13 |   |  0  0  0 60 63 |
    *     \  0  0  0  0 30 /    \  0  0  0  0 14 /   \  0  0  0  0 74 /
    */

   RealType newValue = 1;
   RealType multiplicator = 2;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         if( abs( i - j ) <= 1 )
            m.addElement( i, j, newValue++, multiplicator );

   EXPECT_EQ( m.getElement( 0, 0 ),  3 );
   EXPECT_EQ( m.getElement( 0, 1 ),  6 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ), 15 );
   EXPECT_EQ( m.getElement( 1, 1 ), 18 );
   EXPECT_EQ( m.getElement( 1, 2 ), 21 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 30 );
   EXPECT_EQ( m.getElement( 2, 2 ), 33 );
   EXPECT_EQ( m.getElement( 2, 3 ), 36 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 45  );
   EXPECT_EQ( m.getElement( 3, 3 ), 48 );
   EXPECT_EQ( m.getElement( 3, 4 ), 51 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 60 );
   EXPECT_EQ( m.getElement( 4, 4 ), 63 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 74 );
}

template< typename Matrix >
void test_SetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 3x7 dense matrix:
    *
    *    /  1  2  0  0  0  0  0 \
    *    |  8  9 10  0  0  0  0 |
    *    \  0 16 17 18  0  0  0 /
    */
   const IndexType rows = 3;
   const IndexType cols = 7;

   Matrix m( rows, cols );

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 3 ][ 3 ] {
         {  1,  2,  0 },
         {  8,  9, 10 },
         { 16, 17, 18 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 3; i++ )
      {
         if( rowIdx == 0 && i > 1 )
            break;
         row.setElement( i, values[ rowIdx ][ i ] );
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( 0, 3, f );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  8 );
   EXPECT_EQ( m.getElement( 1, 1 ),  9 );
   EXPECT_EQ( m.getElement( 1, 2 ), 10 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 16 );
   EXPECT_EQ( m.getElement( 2, 2 ), 17 );
   EXPECT_EQ( m.getElement( 2, 3 ), 18 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );
   EXPECT_EQ( m.getElement( 2, 6 ),  0 );
}

template< typename Matrix >
void test_AddRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  0  0  0 \
    *    |  6  7  8  0  0 |
    *    |  0 12 13 14  0 |
    *    |  0  0 18 19 20 |
    *    |  0  0  0 24 25 |
    *    \  0  0  0  0 30 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
      {
         if( abs( i - j ) <= 1 )
            m.setElement( i, j, value );
         value++;
      }

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 30 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *  / 0  0  0  0  0  0 \   /  1  2  0  0  0 \   / 11 11  0  0  0 \   / 11  11  0   0   0 \
    *  | 0  1  0  0  0  0 |   |  6  7  8  0  0 |   | 22 22 22  0  0 |   | 28  29 30   0   0 |
    *  | 0  0  2  0  0  0 | * |  0 12 13 14  0 | + |  0 33 33 33  0 | = |  0  57 59  61   0 |
    *  | 0  0  0  3  0  0 |   |  0  0 18 19 20 |   |  0  0 44 44 44 |   |  0   0 98 101 104 |
    *  | 0  0  0  0  4  0 |   |  0  0  0 24 25 |   |  0  0  0 55 55 |   |  0   0  0 151 155 |
    *  \ 0  0  0  0  0  5 /   \  0  0  0  0 30 /   \  0  0  0  0 66 /   \  0   0  0   0 216 /
    */

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 6 ][ 3 ] {
         { 11, 11,  0 },
         { 22, 22, 22 },
         { 33, 33, 33 },
         { 44, 44, 44 },
         { 55, 55, 55 },
         { 66, 66, 66 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 3; i++ )
      {
         RealType& val = row.getValue( i );
         val = rowIdx * val + values[ rowIdx ][ i ];
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( 0, 6, f );


   EXPECT_EQ( m.getElement( 0, 0 ),  11 );
   EXPECT_EQ( m.getElement( 0, 1 ),  11 );
   EXPECT_EQ( m.getElement( 0, 2 ),   0 );
   EXPECT_EQ( m.getElement( 0, 3 ),   0 );
   EXPECT_EQ( m.getElement( 0, 4 ),   0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  28 );
   EXPECT_EQ( m.getElement( 1, 1 ),  29 );
   EXPECT_EQ( m.getElement( 1, 2 ),  30 );
   EXPECT_EQ( m.getElement( 1, 3 ),   0 );
   EXPECT_EQ( m.getElement( 1, 4 ),   0 );

   EXPECT_EQ( m.getElement( 2, 0 ),   0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  57 );
   EXPECT_EQ( m.getElement( 2, 2 ),  59 );
   EXPECT_EQ( m.getElement( 2, 3 ),  61 );
   EXPECT_EQ( m.getElement( 2, 4 ),   0  );

   EXPECT_EQ( m.getElement( 3, 0 ),   0 );
   EXPECT_EQ( m.getElement( 3, 1 ),   0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  98 );
   EXPECT_EQ( m.getElement( 3, 3 ), 101 );
   EXPECT_EQ( m.getElement( 3, 4 ), 104 );

   EXPECT_EQ( m.getElement( 4, 0 ),   0 );
   EXPECT_EQ( m.getElement( 4, 1 ),   0 );
   EXPECT_EQ( m.getElement( 4, 2 ),   0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 151 );
   EXPECT_EQ( m.getElement( 4, 4 ), 155 );

   EXPECT_EQ( m.getElement( 5, 0 ),   0 );
   EXPECT_EQ( m.getElement( 5, 1 ),   0 );
   EXPECT_EQ( m.getElement( 5, 2 ),   0 );
   EXPECT_EQ( m.getElement( 5, 3 ),   0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 216 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    |  0  0 15 16 |
    *    \  0  0  0 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
      {
         if( abs( i - j ) <= 1 )
            m.setElement( i, j, value );
         value++;
      }

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   VectorType inVector( 4 );
   inVector = 2;

   VectorType outVector( 5 );
   outVector = 0;

   m.vectorProduct( inVector, outVector);

   EXPECT_EQ( outVector.getElement( 0 ),  6 );
   EXPECT_EQ( outVector.getElement( 1 ), 36 );
   EXPECT_EQ( outVector.getElement( 2 ), 66 );
   EXPECT_EQ( outVector.getElement( 3 ), 62 );
   EXPECT_EQ( outVector.getElement( 4 ), 40 );
}

template< typename Matrix1, typename Matrix2 = Matrix1 >
void test_AddMatrix()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    |  0  0 15 16 |
    *    \  0  0  0 20 /
    */
   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix1 m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
      {
         if( abs( i - j ) <= 1 )
            m.setElement( i, j, value );
         value++;
      }

   /*
    * Sets up the following 5x4 dense matrix:
    *
    *    /  1  2  0  0 \
    *    |  3  4  5  0 |
    *    |  0  6  7  8 |
    *    |  0  0  9 10 |
    *    \  0  0  0 11 /
    */
   Matrix2 m2( rows, cols );

   RealType newValue = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++)
         if( abs( i - j ) <= 1 )
            m2.setElement( i, j, newValue++ );

   /*
    * Compute the following 5x4 dense matrix:
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
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const IndexType leftRows = 5;
    const IndexType leftCols = 4;

    Matrix leftMatrix;
    leftMatrix.reset();
    leftMatrix.setDimensions( leftRows, leftCols );

    RealType value = 1;
    for( IndexType i = 0; i < leftRows; i++ )
        for( IndexType j = 0; j < leftCols; j++)
            leftMatrix.setElement( i, j, value++ );

/*
 * Sets up the following 4x5 dense matrix:
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
 * Sets up the following 5x5 resulting dense matrix:
 *
 *    /  0  0  0  0 \
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    \  0  0  0  0 /
 */

    Matrix mResult;
    mResult.reset();
    mResult.setDimensions( leftRows, rightCols );
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
/*
 * Sets up the following 3x2 dense matrix:
 *
 *    /  1  2 \
 *    |  3  4 |
 *    \  5  6 /
 */
    const IndexType rows = 3;
    const IndexType cols = 2;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );

    m.print( std::cout );

/*
 * Sets up the following 2x3 dense matrix:
 *
 *    /  0  0  0 \
 *    \  0  0  0 /
 */
    Matrix mTransposed;
    mTransposed.reset();
    mTransposed.setDimensions( cols, rows );

    mTransposed.print( std::cout );

    RealType matrixMultiplicator = 1;

    mTransposed.getTransposition( m, matrixMultiplicator );

    mTransposed.print( std::cout );

/*
 * Should result in the following 2x3 dense matrix:
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
/*
 * Sets up the following 4x4 dense matrix:
 *
 *    /  4  1  1  1 \
 *    |  1  4  1  1 |
 *    |  1  1  4  1 |
 *    \  1  1  1  4 /
 */
    const IndexType rows = 4;
    const IndexType cols = 4;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

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
   constexpr bool rowMajorOrder = Matrix::getRowMajorOrder();

   using TridiagonalHost = TNL::Matrices::Tridiagonal< RealType, TNL::Devices::Host, IndexType, rowMajorOrder >;
   using TridiagonalCuda = TNL::Matrices::Tridiagonal< RealType, TNL::Devices::Cuda, IndexType, !rowMajorOrder >;

   const IndexType rows( 10 ), columns( 10 );
   TridiagonalHost hostMatrix( rows, columns );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j <  columns; j++ )
         if( abs( i - j ) <= 1 )
            hostMatrix.setElement( i, j,  i + j );

   Matrix matrix( rows, columns );
   matrix.getValues() = 0.0;
   matrix = hostMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ )
            if( abs( i - j ) <= 1 )
               EXPECT_EQ( matrix.getElement( i, j ), i + j );
            else
               EXPECT_EQ( matrix.getElement( i, j ), 0.0 );

#ifdef HAVE_CUDA
   TridiagonalCuda cudaMatrix( rows, columns );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
         if( abs( i - j ) <= 1 )
            cudaMatrix.setElement( i, j, i + j );

   matrix.getValues() = 0.0;
   matrix = cudaMatrix;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( abs( i - j ) <= 1 )
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

   /*
    * Sets up the following 4x4 dense matrix:
    *
    *    /  1  2  0  0 \
    *    |  5  6  7  0 |
    *    |  0 10 11 12 |
    *    \  0  0 15 16 /
    */
   const IndexType rows = 4;
   const IndexType cols = 4;

   Matrix savedMatrix( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
      {
         if( abs( i - j ) <= 1 )
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

   Matrix m( rows, cols );

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
    TNL::Matrices::Tridiagonal< int,    TNL::Devices::Host, short >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Host, short >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Host, short >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Host, short >,
    TNL::Matrices::Tridiagonal< int,    TNL::Devices::Host, int >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Host, int >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Host, int >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Host, int >,
    TNL::Matrices::Tridiagonal< int,    TNL::Devices::Host, long >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Host, long >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Host, long >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
    ,TNL::Matrices::Tridiagonal< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::Tridiagonal< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::Tridiagonal< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::Tridiagonal< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::Tridiagonal< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::Tridiagonal< double, TNL::Devices::Cuda, long >
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

TYPED_TEST( MatrixTest, getCompressedRowLengthTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetCompressedRowLengths< MatrixType >();
}

TYPED_TEST( MatrixTest, getRowLengthTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetRowLength< MatrixType >();
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

TYPED_TEST( MatrixTest, addMatrixTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, addMatrixTest_differentOrdering )
{
    using MatrixType = typename TestFixture::MatrixType;

    using RealType = typename MatrixType::RealType;
    using DeviceType = typename MatrixType::DeviceType;
    using IndexType = typename MatrixType::IndexType;
    using RealAllocatorType = typename MatrixType::RealAllocatorType;
    using MatrixType2 = TNL::Matrices::Tridiagonal< RealType, DeviceType, IndexType, ! MatrixType::getRowMajorOrder(), RealAllocatorType >;

    test_AddMatrix< MatrixType, MatrixType2 >();
}

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

//// test_getType is not general enough yet. DO NOT TEST IT YET.

//TEST( TridiagonalMatrixTest, Tridiagonal_GetTypeTest_Host )
//{
//    host_test_GetType< Tridiagonal_host_float, Tridiagonal_host_int >();
//}
//
//#ifdef HAVE_CUDA
//TEST( TridiagonalMatrixTest, Tridiagonal_GetTypeTest_Cuda )
//{
//    cuda_test_GetType< Tridiagonal_cuda_float, Tridiagonal_cuda_int >();
//}
//#endif

/*TEST( TridiagonalMatrixTest, Tridiagonal_getMatrixProductTest_Host )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h(609): error: no instance of function template \"TNL::Matrices::TridiagonalMatrixProductKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Tridiagonal<int, TNL::Devices::Host, int> *, Tridiagonal_host_int *, Tridiagonal_host_int *, const int, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Tridiagonal<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix1=Tridiagonal_host_int, Matrix2=Tridiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Tridiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(1315): here\n\n";
}

#ifdef HAVE_CUDA
TEST( TridiagonalMatrixTest, Tridiagonal_getMatrixProductTest_Cuda )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h(510): error: identifier \"tnlCudaMin\" is undefined\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::TridiagonalMatrixProductKernel<Real,Index,Matrix1,Matrix2,tileDim,tileRowBlockSize>(TNL::Matrices::Tridiagonal<Real, TNL::Devices::Cuda, Index> *, const Matrix1 *, const Matrix2 *, Real, Real, Index, Index) [with Real=int, Index=int, Matrix1=Tridiagonal_cuda_int, Matrix2=Tridiagonal_cuda_int, tileDim=32, tileRowBlockSize=8]\"\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Tridiagonal<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Cuda, Index=int, Matrix1=Tridiagonal_cuda_int, Matrix2=Tridiagonal_cuda_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Tridiagonal_cuda_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(1332): here\n\n";
}
#endif

TEST( TridiagonalMatrixTest, Tridiagonal_getTranspositionTest_Host )
{
//    test_GetTransposition< Tridiagonal_host_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h(836): error: no instance of function template \"TNL::Matrices::TridiagonalTranspositionAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Tridiagonal<int, TNL::Devices::Host, int> *, Tridiagonal_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Tridiagonal<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Tridiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Tridiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(1420): here\n\n";
    std::cout << "AND this message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h(852): error: no instance of function template \"TNL::Matrices::TridiagonalTranspositionNonAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Tridiagonal<int, TNL::Devices::Host, int> *, Tridiagonal_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Tridiagonal<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Tridiagonal<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Tridiagonal_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Tridiagonal_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/TridiagonalMatrixTest.h(1420): here\n\n";
}

#ifdef HAVE_CUDA
TEST( TridiagonalMatrixTest, Tridiagonal_getTranspositionTest_Cuda )
{
//    test_GetTransposition< Tridiagonal_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test throws the following message: \n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!!\n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Tridiagonal_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  terminate called after throwing an instance of 'TNL::Exceptions::CudaRuntimeError'\n";
    std::cout << "          what():  CUDA ERROR 4 (cudaErrorLaunchFailure): unspecified launch failure.\n";
    std::cout << "  Source: line 57 in /home/lukas/tnl-dev/src/TNL/Containers/Algorithms/ArrayOperationsCuda_impl.h: unspecified launch failure\n";
    std::cout << "  [1]    4003 abort (core dumped)  ./TridiagonalMatrixTest-dbg\n";
}
#endif

TEST( TridiagonalMatrixTest, Tridiagonal_performSORIterationTest_Host )
{
    test_PerformSORIteration< Tridiagonal_host_float >();
}

#ifdef HAVE_CUDA
TEST( TridiagonalMatrixTest, Tridiagonal_performSORIterationTest_Cuda )
{
//    test_PerformSORIteration< Tridiagonal_cuda_float >();
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
