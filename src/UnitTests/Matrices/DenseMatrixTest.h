/***************************************************************************
                          DenseMatrixTest.h -  description
                             -------------------
    begin                : Nov 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// TODO
/*
  * getType()                           ::HOW?  How to test this for each format? edit string how?
 *      MISTAKE! found it for Cuda instead of Devices::Cuda. Incorrect String in src/TNL/Devices/Cuda.cpp
 * getTypeVirtual()                     ::TEST? This just calls getType().
 * getSerializationType()               ::TEST? This just calls getType().
 * getSerializationTypeVirtual()        ::TEST? This just calls getSerializationType().
 * setDimensions()                          ::DONE
 * setLike()                                ::DONE
 * setCompressedRowLengths()            ::NOT IMPLEMENTED! The function body is empty.
 * getRowLength()                           ::DONE
 * getRowLengthFast()                   ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * getMaxRowLength()                    ::TEST? This function is identical to getRowLength().
 * getNumberOfMatrixElements()              ::DONE
 * getNumberOfNonZeroMatrixElements()       ::DONE
 * reset()                                  ::DONE
 * setValue()                               ::DONE
 * operator()                           ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * const operator()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setElementFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setElement()                             ::DONE ; USED! in any test with individual value assignment.
 * addElementFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addElement()                             ::DONE
 * setRowFast()                         ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setRow()                                 ::DONE
 *      MISTAKE! This function unlike the setRow() for CSR, doesn't replace all the elements of a row, it only replaces the elements it has values for in its arrays.
 * addRowFast()                         ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addRow()                                 ::DONE
 * getElementFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * getElement()                             ::USED! in any test with individual value reading.
 * getRowFast()                         ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * getRow()                             ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * const getRow()                       ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * MatrixRow getRow()                   ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * ConstMatrixRow getRow()              ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * rowVectorProduct()                   ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * vectorProduct()                      ::HOW? Throwing abort, CUDA illegal memory access errors.
 * addMatrix()                              ::DONE
 * DenseMatrixProductKernel()           ::HOW? How to test __global__?
 * getMatrixProdut()                    ::HOW? It won't build: When testing CPU: no parameters match function DenseMatrixProductKernel(); when testing GPU: identifier tnlCudaMin is undefined. 
 * DenseTranspositionAlignedKernel()    ::HOW? How to test __global__?
 * DenseTranspositionNonAlignedKernel() ::HOW? How to test __global__?
 * getTransposition()                   ::HOW? It won't build when testing CPU: no parameters match functions DenseTranspositionAlignedKernel() and DenseTranspositionNonAlignedKernel(). On GPU if will throw terminate and (core dumped).
 *      MISTAKE! For GPU it works completely fine, when rows == cols. Otherwise it throws assertion failed.
 * performSORIteration()                ::HOW? Throws segmentation fault CUDA.
 * operator=()                          ::HOW? What is this supposed to enable? Overloading operators?
 * save( String& fileName )                 ::DONE
 * load( String& fileName )                 ::DONE
 * save( File& file)                    ::USED! In save( String& fileName )
 * load( File& file )                   ::USED! In load( String& fileName )
 * print()                                  ::DONE
 * getElementIndex()                    ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 */

// GENERAL TODO
/*
 * Figure out __cuda_callable_. When trying to call __cuda_callable__ functions
 *      a segmentation fault (core dumped) is thrown.
 *  ==>__cuda_callable__ works only for CPU at the moment. (for loops vs thread kernel assignment)
 */


#include <TNL/Devices/Host.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/Dense.h>
#include <TNL/Containers/Array.h>

#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

using Dense_host_float = TNL::Matrices::Dense< float, TNL::Devices::Host, int >;
using Dense_host_int = TNL::Matrices::Dense< int, TNL::Devices::Host, int >;

using Dense_cuda_float = TNL::Matrices::Dense< float, TNL::Devices::Cuda, int >;
using Dense_cuda_int = TNL::Matrices::Dense< int, TNL::Devices::Cuda, int >;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>


template< typename MatrixHostFloat, typename MatrixHostInt >
void host_test_GetType()
{
    MatrixHostFloat mtrxHostFloat;
    MatrixHostInt mtrxHostInt;
    
    EXPECT_EQ( mtrxHostFloat.getType(), TNL::String( "Matrices::Dense< float, Devices::Host, int >" ) );
    EXPECT_EQ( mtrxHostInt.getType(), TNL::String( "Matrices::Dense< int, Devices::Host, int >" ) );
}

// QUESITON: Cant these two functions be combined into one? Because if no CUDA is present and we were to call
//           CUDA into the function in the TEST, to be tested, then we could have a problem.

template< typename MatrixCudaFloat, typename MatrixCudaInt >
void cuda_test_GetType()
{
    MatrixCudaFloat mtrxCudaFloat;
    MatrixCudaInt mtrxCudaInt;

    EXPECT_EQ( mtrxCudaFloat.getType(), TNL::String( "Matrices::Dense< float, Cuda, int >" ) );    // This is mistakenly labeled in /src/TNL/Devices/Cuda.cpp
    EXPECT_EQ( mtrxCudaInt.getType(), TNL::String( "Matrices::Dense< int, Cuda, int >" ) );        // Should be Devices::Cuda
}

template< typename Matrix >
void test_SetDimensions()
{
    const int rows = 9;
    const int cols = 8;
    
    Matrix m;
    m.setDimensions( rows, cols );
    
    EXPECT_EQ( m.getRows(), 9 );
    EXPECT_EQ( m.getColumns(), 8 );
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
void test_GetRowLength()
{
    const int rows = 8;
    const int cols = 7;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    EXPECT_EQ( m.getRowLength( 0 ), 7 );
    EXPECT_EQ( m.getRowLength( 1 ), 7 );
    EXPECT_EQ( m.getRowLength( 2 ), 7 );
    EXPECT_EQ( m.getRowLength( 3 ), 7 );
    EXPECT_EQ( m.getRowLength( 4 ), 7 );
    EXPECT_EQ( m.getRowLength( 5 ), 7 );
    EXPECT_EQ( m.getRowLength( 6 ), 7 );
    EXPECT_EQ( m.getRowLength( 7 ), 7 );
}

template< typename Matrix >
void test_GetNumberOfMatrixElements()
{
    const int rows = 7;
    const int cols = 6;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    EXPECT_EQ( m.getNumberOfMatrixElements(), 42 );
}

template< typename Matrix >
void test_GetNumberOfNonzeroMatrixElements()
{
/*
 * Sets up the following 7x6 dense matrix:
 *
 *    /  0  2  3  4  5  6 \
 *    |  7  8  9 10 11 12 |
 *    | 13 14 15 16 17 18 |
 *    | 19 20 21 22 23 24 |
 *    | 25 26 27 28 29 30 |
 *    | 31 32 33 34 35 36 |
 *    \ 37 38 39 40 41  0 /
 */
    const int rows = 7;
    const int cols = 6;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
    
    m.setElement( 0, 0, 0); // Set the first element of the diagonal to 0.
    m.setElement( 6, 5, 0); // Set the last element of the diagonal to 0.
    
    EXPECT_EQ( m.getNumberOfNonzeroMatrixElements(), 40 );
}

template< typename Matrix >
void test_Reset()
{
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  0  0  0  0 \
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    \  0  0  0  0 /
 */
    const int rows = 5;
    const int cols = 4;
    
    Matrix m;
    m.setDimensions( rows, cols );
    
    m.reset();
    
    EXPECT_EQ( m.getRows(), 0 );
    EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_SetValue()
{
/*
 * Sets up the following 7x6 dense matrix:
 *
 *    /  0  2  3  4  5  6 \
 *    |  7  8  9 10 11 12 |
 *    | 13 14 15 16 17 18 |
 *    | 19 20 21 22 23 24 |
 *    | 25 26 27 28 29 30 |
 *    | 31 32 33 34 35 36 |
 *    \ 37 38 39 40 41  0 /
 */
    const int rows = 7;
    const int cols = 6;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
    
    // Set the values of all elements to a certain number
    m.setValue( 42 );
    
    EXPECT_EQ( m.getElement( 0, 0 ), 42 );
    EXPECT_EQ( m.getElement( 0, 1 ), 42 );
    EXPECT_EQ( m.getElement( 0, 2 ), 42 );
    EXPECT_EQ( m.getElement( 0, 3 ), 42 );
    EXPECT_EQ( m.getElement( 0, 4 ), 42 );
    EXPECT_EQ( m.getElement( 0, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 42 );
    EXPECT_EQ( m.getElement( 1, 1 ), 42 );
    EXPECT_EQ( m.getElement( 1, 2 ), 42 );
    EXPECT_EQ( m.getElement( 1, 3 ), 42 );
    EXPECT_EQ( m.getElement( 1, 4 ), 42 );
    EXPECT_EQ( m.getElement( 1, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 42 );
    EXPECT_EQ( m.getElement( 2, 1 ), 42 );
    EXPECT_EQ( m.getElement( 2, 2 ), 42 );
    EXPECT_EQ( m.getElement( 2, 3 ), 42 );
    EXPECT_EQ( m.getElement( 2, 4 ), 42 );
    EXPECT_EQ( m.getElement( 2, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 42 );
    EXPECT_EQ( m.getElement( 3, 1 ), 42 );
    EXPECT_EQ( m.getElement( 3, 2 ), 42 );
    EXPECT_EQ( m.getElement( 3, 3 ), 42 );
    EXPECT_EQ( m.getElement( 3, 4 ), 42 );
    EXPECT_EQ( m.getElement( 3, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 42 );
    EXPECT_EQ( m.getElement( 4, 1 ), 42 );
    EXPECT_EQ( m.getElement( 4, 2 ), 42 );
    EXPECT_EQ( m.getElement( 4, 3 ), 42 );
    EXPECT_EQ( m.getElement( 4, 4 ), 42 );
    EXPECT_EQ( m.getElement( 4, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 42 );
    EXPECT_EQ( m.getElement( 5, 1 ), 42 );
    EXPECT_EQ( m.getElement( 5, 2 ), 42 );
    EXPECT_EQ( m.getElement( 5, 3 ), 42 );
    EXPECT_EQ( m.getElement( 5, 4 ), 42 );
    EXPECT_EQ( m.getElement( 5, 5 ), 42 );
    
    EXPECT_EQ( m.getElement( 6, 0 ), 42 );
    EXPECT_EQ( m.getElement( 6, 1 ), 42 );
    EXPECT_EQ( m.getElement( 6, 2 ), 42 );
    EXPECT_EQ( m.getElement( 6, 3 ), 42 );
    EXPECT_EQ( m.getElement( 6, 4 ), 42 );
    EXPECT_EQ( m.getElement( 6, 5 ), 42 );
}

template< typename Matrix >
void test_SetElement()
{
/*
 * Sets up the following 5x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    | 16 17 18 19 20 |
 *    \ 21 22 23 24 25 /
 */
    const int rows = 5;
    const int cols = 5;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );    
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
    
    EXPECT_EQ( m.getElement( 0, 0 ),  1 );
    EXPECT_EQ( m.getElement( 0, 1 ),  2 );
    EXPECT_EQ( m.getElement( 0, 2 ),  3 );
    EXPECT_EQ( m.getElement( 0, 3 ),  4 );
    EXPECT_EQ( m.getElement( 0, 4 ),  5 );
    
    EXPECT_EQ( m.getElement( 1, 0 ),  6 );
    EXPECT_EQ( m.getElement( 1, 1 ),  7 );
    EXPECT_EQ( m.getElement( 1, 2 ),  8 );
    EXPECT_EQ( m.getElement( 1, 3 ),  9 );
    EXPECT_EQ( m.getElement( 1, 4 ), 10 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 11 );
    EXPECT_EQ( m.getElement( 2, 1 ), 12 );
    EXPECT_EQ( m.getElement( 2, 2 ), 13 );
    EXPECT_EQ( m.getElement( 2, 3 ), 14 );
    EXPECT_EQ( m.getElement( 2, 4 ), 15 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 16 );
    EXPECT_EQ( m.getElement( 3, 1 ), 17 );
    EXPECT_EQ( m.getElement( 3, 2 ), 18 );
    EXPECT_EQ( m.getElement( 3, 3 ), 19 );
    EXPECT_EQ( m.getElement( 3, 4 ), 20 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 21 );
    EXPECT_EQ( m.getElement( 4, 1 ), 22 );
    EXPECT_EQ( m.getElement( 4, 2 ), 23 );
    EXPECT_EQ( m.getElement( 4, 3 ), 24 );
    EXPECT_EQ( m.getElement( 4, 4 ), 25 );
}

template< typename Matrix >
void test_AddElement()
{
/*
 * Sets up the following 6x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    | 16 17 18 19 20 |
 *    | 21 22 23 24 25 |
 *    \ 26 27 28 29 30 /
 */
    const int rows = 6;
    const int cols = 5;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
    
    // Check the added elements
    EXPECT_EQ( m.getElement( 0, 0 ),  1 );
    EXPECT_EQ( m.getElement( 0, 1 ),  2 );
    EXPECT_EQ( m.getElement( 0, 2 ),  3 );
    EXPECT_EQ( m.getElement( 0, 3 ),  4 );
    EXPECT_EQ( m.getElement( 0, 4 ),  5 );
    
    EXPECT_EQ( m.getElement( 1, 0 ),  6 );
    EXPECT_EQ( m.getElement( 1, 1 ),  7 );
    EXPECT_EQ( m.getElement( 1, 2 ),  8 );
    EXPECT_EQ( m.getElement( 1, 3 ),  9 );
    EXPECT_EQ( m.getElement( 1, 4 ), 10 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 11 );
    EXPECT_EQ( m.getElement( 2, 1 ), 12 );
    EXPECT_EQ( m.getElement( 2, 2 ), 13 );
    EXPECT_EQ( m.getElement( 2, 3 ), 14 );
    EXPECT_EQ( m.getElement( 2, 4 ), 15 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 16 );
    EXPECT_EQ( m.getElement( 3, 1 ), 17 );
    EXPECT_EQ( m.getElement( 3, 2 ), 18 );
    EXPECT_EQ( m.getElement( 3, 3 ), 19 );
    EXPECT_EQ( m.getElement( 3, 4 ), 20 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 21 );
    EXPECT_EQ( m.getElement( 4, 1 ), 22 );
    EXPECT_EQ( m.getElement( 4, 2 ), 23 );
    EXPECT_EQ( m.getElement( 4, 3 ), 24 );
    EXPECT_EQ( m.getElement( 4, 4 ), 25 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 26 );
    EXPECT_EQ( m.getElement( 5, 1 ), 27 );
    EXPECT_EQ( m.getElement( 5, 2 ), 28 );
    EXPECT_EQ( m.getElement( 5, 3 ), 29 );
    EXPECT_EQ( m.getElement( 5, 4 ), 30 );
    
    // Add new elements to the old elements with a multiplying factor applied to the old elements.
/*
 * The following setup results in the following 6x5 dense matrix:
 *
 *    /  3  6  9 12 15 \
 *    | 18 21 24 27 30 |
 *    | 33 36 39 42 45 |
 *    | 48 51 54 57 60 |
 *    | 63 66 69 72 75 |
 *    \ 78 81 84 87 90 /
 */
    int newValue = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.addElement( i, j, newValue++, 2.0 );    
    
    EXPECT_EQ( m.getElement( 0, 0 ),  3 );
    EXPECT_EQ( m.getElement( 0, 1 ),  6 );
    EXPECT_EQ( m.getElement( 0, 2 ),  9 );
    EXPECT_EQ( m.getElement( 0, 3 ), 12 );
    EXPECT_EQ( m.getElement( 0, 4 ), 15 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 18 );
    EXPECT_EQ( m.getElement( 1, 1 ), 21 );
    EXPECT_EQ( m.getElement( 1, 2 ), 24 );
    EXPECT_EQ( m.getElement( 1, 3 ), 27 );
    EXPECT_EQ( m.getElement( 1, 4 ), 30 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 33 );
    EXPECT_EQ( m.getElement( 2, 1 ), 36 );
    EXPECT_EQ( m.getElement( 2, 2 ), 39 );
    EXPECT_EQ( m.getElement( 2, 3 ), 42 );
    EXPECT_EQ( m.getElement( 2, 4 ), 45 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 48 );
    EXPECT_EQ( m.getElement( 3, 1 ), 51 );
    EXPECT_EQ( m.getElement( 3, 2 ), 54 );
    EXPECT_EQ( m.getElement( 3, 3 ), 57 );
    EXPECT_EQ( m.getElement( 3, 4 ), 60 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 63 );
    EXPECT_EQ( m.getElement( 4, 1 ), 66 );
    EXPECT_EQ( m.getElement( 4, 2 ), 69 );
    EXPECT_EQ( m.getElement( 4, 3 ), 72 );
    EXPECT_EQ( m.getElement( 4, 4 ), 75 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 78 );
    EXPECT_EQ( m.getElement( 5, 1 ), 81 );
    EXPECT_EQ( m.getElement( 5, 2 ), 84 );
    EXPECT_EQ( m.getElement( 5, 3 ), 87 );
    EXPECT_EQ( m.getElement( 5, 4 ), 90 );
}

template< typename Matrix >
void test_SetRow()
{
/*
 * Sets up the following 3x7 dense matrix:
 *
 *    /  1  2  3  4  5  6  7 \
 *    |  8  9 10 11 12 13 14 |
 *    \ 15 16 17 18 19 20 21 /
 */
    const int rows = 3;
    const int cols = 7;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
        
    
    int row1 [ 5 ] = { 11, 11, 11, 11, 11 }; int colIndexes1 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row2 [ 5 ] = { 22, 22, 22, 22, 22 }; int colIndexes2 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row3 [ 5 ] = { 33, 33, 33, 33, 33 }; int colIndexes3 [ 5 ] = { 2, 3, 4, 5, 6 };
    
    m.setRow( 0, colIndexes1, row1, 5 );
    m.setRow( 1, colIndexes2, row2, 5 );
    m.setRow( 2, colIndexes3, row3, 5 );
    
    EXPECT_EQ( m.getElement( 0, 0 ), 11 );
    EXPECT_EQ( m.getElement( 0, 1 ), 11 );
    EXPECT_EQ( m.getElement( 0, 2 ), 11 );
    EXPECT_EQ( m.getElement( 0, 3 ), 11 );
    EXPECT_EQ( m.getElement( 0, 4 ), 11 );
    EXPECT_EQ( m.getElement( 0, 5 ),  6 );
    EXPECT_EQ( m.getElement( 0, 6 ),  7 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 22 );
    EXPECT_EQ( m.getElement( 1, 1 ), 22 );
    EXPECT_EQ( m.getElement( 1, 2 ), 22 );
    EXPECT_EQ( m.getElement( 1, 3 ), 22 );
    EXPECT_EQ( m.getElement( 1, 4 ), 22 );
    EXPECT_EQ( m.getElement( 1, 5 ), 13 );
    EXPECT_EQ( m.getElement( 1, 6 ), 14 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 15 );
    EXPECT_EQ( m.getElement( 2, 1 ), 16 );
    EXPECT_EQ( m.getElement( 2, 2 ), 33 );
    EXPECT_EQ( m.getElement( 2, 3 ), 33 );
    EXPECT_EQ( m.getElement( 2, 4 ), 33 );
    EXPECT_EQ( m.getElement( 2, 5 ), 33 );
    EXPECT_EQ( m.getElement( 2, 6 ), 33 );
}

template< typename Matrix >
void test_AddRow()
{
/*
 * Sets up the following 6x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    | 16 17 18 19 20 |
 *    | 21 22 23 24 25 |
 *    \ 26 27 28 29 30 /
 */
    const int rows = 6;
    const int cols = 5;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );
    
    // Check the added elements
    EXPECT_EQ( m.getElement( 0, 0 ),  1 );
    EXPECT_EQ( m.getElement( 0, 1 ),  2 );
    EXPECT_EQ( m.getElement( 0, 2 ),  3 );
    EXPECT_EQ( m.getElement( 0, 3 ),  4 );
    EXPECT_EQ( m.getElement( 0, 4 ),  5 );
    
    EXPECT_EQ( m.getElement( 1, 0 ),  6 );
    EXPECT_EQ( m.getElement( 1, 1 ),  7 );
    EXPECT_EQ( m.getElement( 1, 2 ),  8 );
    EXPECT_EQ( m.getElement( 1, 3 ),  9 );
    EXPECT_EQ( m.getElement( 1, 4 ), 10 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 11 );
    EXPECT_EQ( m.getElement( 2, 1 ), 12 );
    EXPECT_EQ( m.getElement( 2, 2 ), 13 );
    EXPECT_EQ( m.getElement( 2, 3 ), 14 );
    EXPECT_EQ( m.getElement( 2, 4 ), 15 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 16 );
    EXPECT_EQ( m.getElement( 3, 1 ), 17 );
    EXPECT_EQ( m.getElement( 3, 2 ), 18 );
    EXPECT_EQ( m.getElement( 3, 3 ), 19 );
    EXPECT_EQ( m.getElement( 3, 4 ), 20 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 21 );
    EXPECT_EQ( m.getElement( 4, 1 ), 22 );
    EXPECT_EQ( m.getElement( 4, 2 ), 23 );
    EXPECT_EQ( m.getElement( 4, 3 ), 24 );
    EXPECT_EQ( m.getElement( 4, 4 ), 25 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 26 );
    EXPECT_EQ( m.getElement( 5, 1 ), 27 );
    EXPECT_EQ( m.getElement( 5, 2 ), 28 );
    EXPECT_EQ( m.getElement( 5, 3 ), 29 );
    EXPECT_EQ( m.getElement( 5, 4 ), 30 );
    
    // Add new elements to the old elements with a multiplying factor applied to the old elements.
/*
 * The following setup results in the following 6x5 sparse matrix:
 *
 *    /  3  6  9 12 15 \
 *    | 18 21 24 27 30 |
 *    | 33 36 39 42 45 |
 *    | 48 51 54 57 60 |
 *    | 63 66 69 72 75 |
 *    \ 78 81 84 87 90 /
 */
    
    int row0 [ 5 ] = { 11, 11, 11, 11, 0 }; int colIndexes0 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row1 [ 5 ] = { 22, 22, 22, 22, 0 }; int colIndexes1 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row2 [ 5 ] = { 33, 33, 33, 33, 0 }; int colIndexes2 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row3 [ 5 ] = { 44, 44, 44, 44, 0 }; int colIndexes3 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row4 [ 5 ] = { 55, 55, 55, 55, 0 }; int colIndexes4 [ 5 ] = { 0, 1, 2, 3, 4 };
    int row5 [ 5 ] = { 66, 66, 66, 66, 0 }; int colIndexes5 [ 5 ] = { 0, 1, 2, 3, 4 };
    
    m.addRow( 0, colIndexes0, row0, 5, 0.0 );
    m.addRow( 1, colIndexes1, row1, 5, 1.0 );
    m.addRow( 2, colIndexes2, row2, 5, 2.0 );
    m.addRow( 3, colIndexes3, row3, 5, 3.0 );
    m.addRow( 4, colIndexes4, row4, 5, 4.0 );
    m.addRow( 5, colIndexes5, row5, 5, 5.0 );
    
    EXPECT_EQ( m.getElement( 0, 0 ),  11 );
    EXPECT_EQ( m.getElement( 0, 1 ),  11 );
    EXPECT_EQ( m.getElement( 0, 2 ),  11 );
    EXPECT_EQ( m.getElement( 0, 3 ),  11 );
    EXPECT_EQ( m.getElement( 0, 4 ),   0 );
    
    EXPECT_EQ( m.getElement( 1, 0 ),  28 );
    EXPECT_EQ( m.getElement( 1, 1 ),  29 );
    EXPECT_EQ( m.getElement( 1, 2 ),  30 );
    EXPECT_EQ( m.getElement( 1, 3 ),  31 );
    EXPECT_EQ( m.getElement( 1, 4 ),  10 );
    
    EXPECT_EQ( m.getElement( 2, 0 ),  55 );
    EXPECT_EQ( m.getElement( 2, 1 ),  57 );
    EXPECT_EQ( m.getElement( 2, 2 ),  59 );
    EXPECT_EQ( m.getElement( 2, 3 ),  61 );
    EXPECT_EQ( m.getElement( 2, 4 ),  30 );
    
    EXPECT_EQ( m.getElement( 3, 0 ),  92 );
    EXPECT_EQ( m.getElement( 3, 1 ),  95 );
    EXPECT_EQ( m.getElement( 3, 2 ),  98 );
    EXPECT_EQ( m.getElement( 3, 3 ), 101 );
    EXPECT_EQ( m.getElement( 3, 4 ),  60 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 139 );
    EXPECT_EQ( m.getElement( 4, 1 ), 143 );
    EXPECT_EQ( m.getElement( 4, 2 ), 147 );
    EXPECT_EQ( m.getElement( 4, 3 ), 151 );
    EXPECT_EQ( m.getElement( 4, 4 ), 100 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 196 );
    EXPECT_EQ( m.getElement( 5, 1 ), 201 );
    EXPECT_EQ( m.getElement( 5, 2 ), 206 );
    EXPECT_EQ( m.getElement( 5, 3 ), 211 );
    EXPECT_EQ( m.getElement( 5, 4 ), 150 );
}

template< typename Matrix >
void test_VectorProduct()
{
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const int rows = 5;
    const int cols = 4;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++)
            m.setElement( i, j, value++ );
    
    #include <TNL/Containers/Vector.h>
    #include <TNL/Containers/VectorView.h>

    using namespace TNL;
    using namespace TNL::Containers;
    using namespace TNL::Containers::Algorithms;

    Vector< int, Devices::Host, int > inVector;
    inVector.setSize( 4 );
    for( int i = 0; i < inVector.getSize(); i++ )        
        inVector.setElement( i, 2 );

    Vector< int, Devices::Host, int > outVector;  
    outVector.setSize( 5 );
    for( int j = 0; j < outVector.getSize(); j++ )
        outVector.setElement( j, 0 );
 
    
    m.vectorProduct( inVector, outVector);
   
    EXPECT_EQ( outVector.getElement( 0 ),  20 );
    EXPECT_EQ( outVector.getElement( 1 ),  52 );
    EXPECT_EQ( outVector.getElement( 2 ),  84 );
    EXPECT_EQ( outVector.getElement( 3 ), 116 );
    EXPECT_EQ( outVector.getElement( 4 ), 148 );
}

template< typename Matrix >
void test_AddMatrix()
{
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const int rows = 5;
    const int cols = 4;
    
    Matrix m;                           // We need this matrix to preserve the values for EXPECT_EQ statements comparing the actual operation;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++)
            m.setElement( i, j, value++ );
    
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    
    Matrix m2;
    m2.reset();
    m2.setDimensions( rows, cols );
    
    int newValue = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++)
            m2.setElement( i, j, newValue++ );
    
    /*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    
    Matrix mResult;
    mResult.reset();
    mResult.setDimensions( rows, cols );
    
    mResult = m;
    
    int matrixMultiplicator = 2;
    int thisMatrixMultiplicator = 1;
    
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
    EXPECT_EQ( mResult.getElement( 0, 2 ),  9 );
    EXPECT_EQ( mResult.getElement( 0, 3 ), 12 );
    
    EXPECT_EQ( mResult.getElement( 1, 0 ), 15 );
    EXPECT_EQ( mResult.getElement( 1, 1 ), 18 );
    EXPECT_EQ( mResult.getElement( 1, 2 ), 21 );
    EXPECT_EQ( mResult.getElement( 1, 3 ), 24 );
    
    EXPECT_EQ( mResult.getElement( 2, 0 ), 27 );
    EXPECT_EQ( mResult.getElement( 2, 1 ), 30 );
    EXPECT_EQ( mResult.getElement( 2, 2 ), 33 );
    EXPECT_EQ( mResult.getElement( 2, 3 ), 36 );
    
    EXPECT_EQ( mResult.getElement( 3, 0 ), 39 );
    EXPECT_EQ( mResult.getElement( 3, 1 ), 42 );
    EXPECT_EQ( mResult.getElement( 3, 2 ), 45 );
    EXPECT_EQ( mResult.getElement( 3, 3 ), 48 );
    
    EXPECT_EQ( mResult.getElement( 4, 0 ), 51 );
    EXPECT_EQ( mResult.getElement( 4, 1 ), 54 );
    EXPECT_EQ( mResult.getElement( 4, 2 ), 57 );
    EXPECT_EQ( mResult.getElement( 4, 3 ), 60 );
}

template< typename Matrix >
void test_GetMatrixProduct()
{
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const int leftRows = 5;
    const int leftCols = 4;
    
    Matrix leftMatrix;
    leftMatrix.reset();
    leftMatrix.setDimensions( leftRows, leftCols );
    
    int value = 1;
    for( int i = 0; i < leftRows; i++ )
        for( int j = 0; j < leftCols; j++)
            leftMatrix.setElement( i, j, value++ );
    
/*
 * Sets up the following 4x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    \ 16 17 18 19 20 /
 */
    const int rightRows = 4;
    const int rightCols = 5;
    
    Matrix rightMatrix;
    rightMatrix.reset();
    rightMatrix.setDimensions( rightRows, rightCols );
    
    int newValue = 1;
    for( int i = 0; i < rightRows; i++ )
        for( int j = 0; j < rightCols; j++)
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
    
    int leftMatrixMultiplicator = 1;
    int rightMatrixMultiplicator = 2;
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
/*
 * Sets up the following 3x2 dense matrix:
 *
 *    /  1  2 \
 *    |  3  4 |
 *    \  5  6 /
 */
    const int rows = 3;
    const int cols = 2;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
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
    
    mTransposed.getTransposition( m, 1.0 );
    
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
    
/*
 * Sets up the following 5x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    | 16 17 18 19 20 |
 *    \ 21 22 23 24 25 /
 */
            //    const int rows = 5;
            //    const int cols = 5;
            //    
            //    Matrix m;
            //    m.reset();
            //    m.setDimensions( rows, cols );
            //    
            //    int value = 1;
            //    for( int i = 0; i < rows; i++ )
            //        for( int j = 0; j < cols; j++)
            //            m.setElement( i, j, value++ );
    
/*
 * Sets up the following 5x5 dense matrix:
 *
 *    /  2 12 22 32 42 \
 *    |  4 14 24 34 44 |
 *    |  6 16 26 36 46 |
 *    |  8 18 28 38 48 |
 *    \ 10 20 30 40 50 /
 */
            //    const int resultRows = cols;
            //    const int resultCols = rows;
            //    
            //    Matrix mResult;
            //    mResult.reset();
            //    mResult.setDimensions( resultRows, resultCols );
            //    mResult.setValue( 0 );
            //    
            //    int matrixMultiplicator = 2;
            //    
            //    mResult.getTransposition( m, matrixMultiplicator );
    
/*
 * Should result in the following 5x5 resulting dense matrix:
 *
 *    /  0  0  0  0  0 \
 *    |  0  0  0  0  0 |
 *    |  0  0  0  0  0 |
 *    |  0  0  0  0  0 |
 *    \  0  0  0  0  0 /
 */
            //    
            //    EXPECT_EQ( mResult.getElement( 0, 0 ),  2 );
            //    EXPECT_EQ( mResult.getElement( 0, 1 ), 12 );
            //    EXPECT_EQ( mResult.getElement( 0, 2 ), 22 );
            //    EXPECT_EQ( mResult.getElement( 0, 3 ), 32 );
            //    EXPECT_EQ( mResult.getElement( 0, 4 ), 42 );
            //    
            //    EXPECT_EQ( mResult.getElement( 1, 0 ),  4 );
            //    EXPECT_EQ( mResult.getElement( 1, 1 ), 14 );
            //    EXPECT_EQ( mResult.getElement( 1, 2 ), 24 );
            //    EXPECT_EQ( mResult.getElement( 1, 3 ), 34 );
            //    EXPECT_EQ( mResult.getElement( 1, 4 ), 44 );
            //    
            //    EXPECT_EQ( mResult.getElement( 2, 0 ),  6 );
            //    EXPECT_EQ( mResult.getElement( 2, 1 ), 16 );
            //    EXPECT_EQ( mResult.getElement( 2, 2 ), 26 );
            //    EXPECT_EQ( mResult.getElement( 2, 3 ), 36 );
            //    EXPECT_EQ( mResult.getElement( 2, 4 ), 46 );
            //    
            //    EXPECT_EQ( mResult.getElement( 3, 0 ),  8 );
            //    EXPECT_EQ( mResult.getElement( 3, 1 ), 18 );
            //    EXPECT_EQ( mResult.getElement( 3, 2 ), 28 );
            //    EXPECT_EQ( mResult.getElement( 3, 3 ), 38 );
            //    EXPECT_EQ( mResult.getElement( 3, 4 ), 48 );
            //    
            //    EXPECT_EQ( mResult.getElement( 4, 0 ), 10 );
            //    EXPECT_EQ( mResult.getElement( 4, 1 ), 20 );
            //    EXPECT_EQ( mResult.getElement( 4, 2 ), 30 );
            //    EXPECT_EQ( mResult.getElement( 4, 3 ), 40 );
            //    EXPECT_EQ( mResult.getElement( 4, 4 ), 50 );
}


template< typename Matrix >
void test_PerformSORIteration()
{
/*
 * Sets up the following 4x4 dense matrix:
 *
 *    /  4  1  1  1 \
 *    |  1  4  1  1 |
 *    |  1  1  4  1 |
 *    \  1  1  1  4 /
 */
    const int rows = 4;
    const int cols = 4;
    
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
    
    float bVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };
    float xVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };
    
    m.performSORIteration( bVector, 0, xVector, 1);
    
    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ],  1.0 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );
    
    m.performSORIteration( bVector, 1, xVector, 1);
    
    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );
    
    m.performSORIteration( bVector, 2, xVector, 1);
    
    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  0.15625 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );
    
    m.performSORIteration( bVector, 3, xVector, 1);
    
    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ], 0.15625 );
    EXPECT_EQ( xVector[ 3 ], 0.3671875 );
}

template< typename Matrix >
void test_SaveAndLoad()
{
/*
 * Sets up the following 4x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    \ 13 14 15 16 /
 */
    const int rows = 4;
    const int cols = 4;
    
    Matrix savedMatrix;
    savedMatrix.reset();
    savedMatrix.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++ )
        for( int j = 0; j < cols; j++ )
            savedMatrix.setElement( i, j, value++ );
        
    savedMatrix.save( "denseMatrixFile" );
    
    Matrix loadedMatrix;
    loadedMatrix.reset();
    loadedMatrix.setDimensions( rows, cols );
    
    loadedMatrix.load( "denseMatrixFile" );
    
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
    EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  4 );
    
    EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  5 );
    EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  6 );
    EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  7 );
    EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  8 );
    
    EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  9 );
    EXPECT_EQ( savedMatrix.getElement( 2, 1 ), 10 );
    EXPECT_EQ( savedMatrix.getElement( 2, 2 ), 11 );
    EXPECT_EQ( savedMatrix.getElement( 2, 3 ), 12 );
    
    EXPECT_EQ( savedMatrix.getElement( 3, 0 ), 13 );
    EXPECT_EQ( savedMatrix.getElement( 3, 1 ), 14 );
    EXPECT_EQ( savedMatrix.getElement( 3, 2 ), 15 );
    EXPECT_EQ( savedMatrix.getElement( 3, 3 ), 16 );
    
    std::cout << "\nThis will create a file called 'denseMatrixFile' (of the matrix created in the test function), in .../tnl-dev/Debug/bin/\n\n";
}

template< typename Matrix >
void test_Print()
{
/*
 * Sets up the following 5x4 sparse matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const int rows = 5;
    const int cols = 4;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    
    int value = 1;
    for( int i = 0; i < rows; i++)
        for( int j = 0; j < cols; j++)
            m.setElement( i, j, value++ );
    
    // This is from: https://stackoverflow.com/questions/5193173/getting-cout-output-to-a-stdstring
    #include <sstream>
    std::stringstream printed;
    std::stringstream couted;
    
    // This is from: https://stackoverflow.com/questions/19485536/redirect-output-of-an-function-printing-to-console-to-string
    //change the underlying buffer and save the old buffer
    auto old_buf = std::cout.rdbuf(printed.rdbuf()); 

    m.print( std::cout ); //all the std::cout goes to ss

    std::cout.rdbuf(old_buf); //reset
    
    //printed << printed.str() << std::endl;
    couted << "Row: 0 ->  Col:0->1	 Col:1->2	 Col:2->3	 Col:3->4\t\n"
              "Row: 1 ->  Col:0->5	 Col:1->6	 Col:2->7	 Col:3->8\t\n"
              "Row: 2 ->  Col:0->9	 Col:1->10	 Col:2->11	 Col:3->12\t\n"
              "Row: 3 ->  Col:0->13	 Col:1->14	 Col:2->15	 Col:3->16\t\n"
              "Row: 4 ->  Col:0->17	 Col:1->18	 Col:2->19	 Col:3->20\t\n";
    
    EXPECT_EQ( printed.str(), couted.str() );
}

//// test_getType is not general enough yet. DO NOT TEST IT YET.

//TEST( DenseMatrixTest, Dense_GetTypeTest_Host )
//{
//    host_test_GetType< Dense_host_float, Dense_host_int >();
//}
//
//#ifdef HAVE_CUDA
//TEST( DenseMatrixTest, Dense_GetTypeTest_Cuda )
//{
//    cuda_test_GetType< Dense_cuda_float, Dense_cuda_int >();
//}
//#endif

TEST( DenseMatrixTest, DeDense_setDimensionsTest_Host )
{
    test_SetDimensions< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_setDimensionsTest_Cuda )
{
    test_SetDimensions< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_setLikeTest_Host )
{
    test_SetLike< Dense_host_int, Dense_host_float >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_setLikeTest_Cuda )
{
    test_SetLike< Dense_cuda_int, Dense_cuda_float >();
}
#endif

TEST( DenseMatrixTest, Dense_getRowLengthTest_Host )
{
    test_GetRowLength< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getRowLengthTest_Cuda )
{
    test_GetRowLength< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_getNumberOfMatrixElementsTest_Host )
{
    test_GetNumberOfMatrixElements< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getNumberOfMatrixElementsTest_Cuda )
{
    test_GetNumberOfMatrixElements< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_getNumberOfNonzeroMatrixElementsTest_Host )
{
    test_GetNumberOfNonzeroMatrixElements< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getNumberOfNonzeroMatrixElementsTest_Cuda )
{
    test_GetNumberOfNonzeroMatrixElements< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_resetTest_Host )
{
    test_Reset< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_resetTest_Cuda )
{
    test_Reset< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_setValueTest_Host )
{
    test_SetValue< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_setValueTest_Cuda )
{
    test_SetValue< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_setElementTest_Host )
{
    test_SetElement< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_setElementTest_Cuda )
{
    test_SetElement< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_addElementTest_Host )
{
    test_AddElement< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_addElementTest_Cuda )
{
    test_AddElement< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_setRowTest_Host )
{
    test_SetRow< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_setRowTest_Cuda )
{
    test_SetRow< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_addRowTest_Host )
{
    test_AddRow< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_addRowTest_Cuda )
{
    test_AddRow< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_vectorProductTest_Host )
{
    test_VectorProduct< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_vectorProductTest_Cuda )
{
//   test_VectorProduct< Dense_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WOKRING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      terminate called after throwing an instance of 'TNL::Exceptions::CudaRuntimeError'\n";
    std::cout << "        what():  CUDA ERROR 77 (cudaErrorIllegalAddress): an illegal memory access was encountered.\n";
    std::cout << "      Source: line 57 in /home/lukas/tnl-dev/src/TNL/Containers/Algorithms/ArrayOperationsCuda_impl.h: an illegal memory access was encountered\n";
    std::cout << "      [1]    22515 abort (core dumped)  ./SparseMatrixTest-dbg\n\n";
}
#endif

TEST( DenseMatrixTest, Dense_addMatrixTest_Host )
{
    test_AddMatrix< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_addMatrixTest_Cuda )
{
    test_AddMatrix< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_getMatrixProductTest_Host )
{
//    test_GetMatrixProduct< Dense_host_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(609): error: no instance of function template \"TNL::Matrices::DenseMatrixProductKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, Dense_host_int *, const int, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix1=Dense_host_int, Matrix2=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1315): here\n\n";
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getMatrixProductTest_Cuda )
{
//    test_GetMatrixProduct< Dense_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(510): error: identifier \"tnlCudaMin\" is undefined\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::DenseMatrixProductKernel<Real,Index,Matrix1,Matrix2,tileDim,tileRowBlockSize>(TNL::Matrices::Dense<Real, TNL::Devices::Cuda, Index> *, const Matrix1 *, const Matrix2 *, Real, Real, Index, Index) [with Real=int, Index=int, Matrix1=Dense_cuda_int, Matrix2=Dense_cuda_int, tileDim=32, tileRowBlockSize=8]\"\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Cuda, Index=int, Matrix1=Dense_cuda_int, Matrix2=Dense_cuda_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Dense_cuda_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1332): here\n\n";
}
#endif

TEST( DenseMatrixTest, Dense_getTranspositionTest_Host )
{
//    test_GetTransposition< Dense_host_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(836): error: no instance of function template \"TNL::Matrices::DenseTranspositionAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1420): here\n\n";
    std::cout << "AND this message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(852): error: no instance of function template \"TNL::Matrices::DenseTranspositionNonAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1420): here\n\n";
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getTranspositionTest_Cuda )
{
//    test_GetTransposition< Dense_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test throws the following message: \n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!!\n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  terminate called after throwing an instance of 'TNL::Exceptions::CudaRuntimeError'\n";
    std::cout << "          what():  CUDA ERROR 4 (cudaErrorLaunchFailure): unspecified launch failure.\n";
    std::cout << "  Source: line 57 in /home/lukas/tnl-dev/src/TNL/Containers/Algorithms/ArrayOperationsCuda_impl.h: unspecified launch failure\n";
    std::cout << "  [1]    4003 abort (core dumped)  ./DenseMatrixTest-dbg\n";
}
#endif

TEST( DenseMatrixTest, Dense_performSORIterationTest_Host )
{
    test_PerformSORIteration< Dense_host_float >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_performSORIterationTest_Cuda )
{
//    test_PerformSORIteration< Dense_cuda_float >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      [1]    6992 segmentation fault (core dumped)  ./SparseMatrixTest-dbg\n\n";
}
#endif

TEST( DenseMatrixTest, Dense_saveAndLoadTest_Host )
{
    test_SaveAndLoad< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_saveAndLoadTest_Cuda )
{
    test_SaveAndLoad< Dense_cuda_int >();
}
#endif

TEST( DenseMatrixTest, Dense_printTest_Host )
{
    test_Print< Dense_host_int >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_printTest_Cuda )
{
    test_Print< Dense_cuda_int >();
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

