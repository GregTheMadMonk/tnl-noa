/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// TODO
/*
 * getType()                        ::HOW?  How to test this for each format? edit string how?
 *      Found the mistake for Cuda instead of Devices::Cuda. Incorrect String in src/TNL/Devices/Cuda.cpp
 *      MISSING: indexType is missing in CSR_impl.h
 * getTypeVirtual()                 ::TEST? This just calls getType().
 * getSerializationType()           ::TEST? This just calls HostType::getType().
 * getSerializationTypeVirtual()    ::TEST? This just calls getSerializationType().
 * setDimensions()                      ::DONE
 * setCompressedRowLengths()            ::DONE
 * getRowLength()                   ::USED! In test_SetCompressedRowLengths() to verify the test itself.
 * getRowLengthFast()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setLike()                            ::DONE
 * reset()                              ::DONE
 * setElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setElement()                         ::DONE
 * addElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addElement()                         ::DONE
 * setRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setRow()                             ::DONE
 * addRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addRow()                         ::NOT IMPLEMENTED! This calls addRowFast() which isn't implemented. Implement? Is it supposed to add an extra row to the matrix or add elements of a row to another row in the matrix?
 * getElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * getElement()                     ::USED! In test_SetElement(), test_AddElement() and test_setRow() to verify the test itself.
 * getRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * MatrixRow getRow()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * ConstMatrixRow getRow()          ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * rowVectorProduct()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * vectorProduct()                      ::DONE
 *      This used to throw illegal memory access, but instead of using ints for vectors, using Types, helped.
 * addMatrix()                      ::NOT IMPLEMENTED!
 * getTransposition()               ::NOT IMPLMENETED!
 * performSORIteration()            ::HOW? Throws segmentation fault CUDA.
 * operator=()                      ::HOW? What is this supposed to enable? Overloading operators?
 * save( File& file)                ::USED! In save( String& fileName )
 * load( File& file )               ::USED! In load( String& fileName )
 * save( String& fileName )             ::DONE
 * load( String& fileName )             ::DONE
 * print()                              ::DONE
 * setCudaKernelType()              ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getCudaKernelType()              ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * setCudaWarpSize()                ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getCudaWarpSize()                ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * setHybridModeSplit()             ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getHybridModeSplit()             ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * spmvCudaVectorized()             ::TEST? How to test __device__?
 * vectorProductCuda()              ::TEST? How to test __device__?
 */

// GENERAL TODO
/*
 * For every function, EXPECT_EQ needs to be done, even for zeros in matrices.
 * Figure out __cuda_callable_. When trying to call __cuda_callable__ functions
 *      a segmentation fault (core dumped) is thrown.
 *  ==>__cuda_callable__ works only for CPU at the moment. (for loops vs thread kernel assignment)
 */


#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>

#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

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

    EXPECT_EQ( mtrxCudaFloat.getType(), TNL::String( "Matrices::CSR< float, Cuda >" ) );    // This is mistakenly labeled in /src/TNL/Devices/Cuda.cpp
    EXPECT_EQ( mtrxCudaInt.getType(), TNL::String( "Matrices::CSR< int, Cuda >" ) );        // Should be Devices::Cuda
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
    
    EXPECT_EQ( m.getRowLength( 0 ), 3 );
    EXPECT_EQ( m.getRowLength( 1 ), 3 );
    EXPECT_EQ( m.getRowLength( 2 ), 1 );
    EXPECT_EQ( m.getRowLength( 3 ), 2 );
    EXPECT_EQ( m.getRowLength( 4 ), 3 );
    EXPECT_EQ( m.getRowLength( 5 ), 4 );
    EXPECT_EQ( m.getRowLength( 6 ), 5 );
    EXPECT_EQ( m.getRowLength( 7 ), 6 );
    EXPECT_EQ( m.getRowLength( 8 ), 7 );
    EXPECT_EQ( m.getRowLength( 9 ), 8 );
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
/*
 * Sets up the following 5x4 sparse matrix:
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
void test_SetElement()
{
/*
 * Sets up the following 5x5 sparse matrix:
 *
 *    /  1  0  0  0  0 \
 *    |  0  2  0  0  0 |
 *    |  0  0  3  0  0 |
 *    |  0  0  0  4  0 |
 *    \  0  0  0  0  5 /
 */
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
    EXPECT_EQ( m.getElement( 0, 1 ), 0 );
    EXPECT_EQ( m.getElement( 0, 2 ), 0 );
    EXPECT_EQ( m.getElement( 0, 3 ), 0 );
    EXPECT_EQ( m.getElement( 0, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 0 );
    EXPECT_EQ( m.getElement( 1, 1 ), 2 );
    EXPECT_EQ( m.getElement( 1, 2 ), 0 );
    EXPECT_EQ( m.getElement( 1, 3 ), 0 );
    EXPECT_EQ( m.getElement( 1, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 0 );
    EXPECT_EQ( m.getElement( 2, 1 ), 0 );
    EXPECT_EQ( m.getElement( 2, 2 ), 3 );
    EXPECT_EQ( m.getElement( 2, 3 ), 0 );
    EXPECT_EQ( m.getElement( 2, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 0 );
    EXPECT_EQ( m.getElement( 3, 1 ), 0 );
    EXPECT_EQ( m.getElement( 3, 2 ), 0 );
    EXPECT_EQ( m.getElement( 3, 3 ), 4 );
    EXPECT_EQ( m.getElement( 3, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 0 );
    EXPECT_EQ( m.getElement( 4, 1 ), 0 );
    EXPECT_EQ( m.getElement( 4, 2 ), 0 );
    EXPECT_EQ( m.getElement( 4, 3 ), 0 );
    EXPECT_EQ( m.getElement( 4, 4 ), 5 );
}

template< typename Matrix >
void test_AddElement()
{
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
    const int rows = 6;
    const int cols = 5;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 3 );
    m.setCompressedRowLengths( rowLengths );
    
    int value = 1;
    for( int i = 0; i < cols - 2; i++ )     // 0th row
        m.setElement( 0, i, value++ );
    
    for( int i = 1; i < cols - 1; i++ )     // 1st row
        m.setElement( 1, i, value++ );
        
    for( int i = 2; i < cols; i++ )         // 2nd row
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
    int newValue = 1;
    for( int i = 0; i < cols - 2; i++ )         // 0th row
        m.addElement( 0, i, newValue++, 2.0 );
    
    for( int i = 1; i < cols - 1; i++ )         // 1st row
        m.addElement( 1, i, newValue++, 2.0 );
        
    for( int i = 2; i < cols; i++ )             // 2nd row
        m.addElement( 2, i, newValue++, 2.0 );
        
    for( int i = 0; i < cols - 2; i++ )         // 3rd row
        m.addElement( 3, i, newValue++, 2.0 );
    
    for( int i = 1; i < cols - 1; i++ )         // 4th row
        m.addElement( 4, i, newValue++, 2.0 );
    
    for( int i = 2; i < cols; i++ )             // 5th row
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
void test_SetRow()
{
/*
 * Sets up the following 3x7 sparse matrix:
 *
 *    /  0  0  0  1  1  1  0 \
 *    |  2  2  2  0  0  0  0 |
 *    \  3  3  3  0  0  0  0 /
 */
    const int rows = 3;
    const int cols = 7;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 6 );
    rowLengths.setElement( 1, 3 );
    m.setCompressedRowLengths( rowLengths );
    
    int value = 1;
    for( int i = 0; i < 3; i++ )
    {
        m.setElement( 0, i + 3, value );
        m.setElement( 1, i, value + 1 );
        m.setElement( 2, i, value + 2);
    }
    
    int row1 [ 3 ] = { 11, 11, 11 }; int colIndexes1 [ 3 ] = { 0, 1, 2 };
    int row2 [ 3 ] = { 22, 22, 22 }; int colIndexes2 [ 3 ] = { 0, 1, 2 };
    int row3 [ 3 ] = { 33, 33, 33 }; int colIndexes3 [ 3 ] = { 3, 4, 5 };
    
    m.setRow( 0, colIndexes1, row1, 3 );
    m.setRow( 1, colIndexes2, row2, 3 );
    m.setRow( 2, colIndexes3, row3, 3 );
    
    EXPECT_EQ( m.getElement( 0, 0 ), 11 );
    EXPECT_EQ( m.getElement( 0, 1 ), 11 );
    EXPECT_EQ( m.getElement( 0, 2 ), 11 );
    EXPECT_EQ( m.getElement( 0, 3 ),  0 );
    EXPECT_EQ( m.getElement( 0, 4 ),  0 );
    EXPECT_EQ( m.getElement( 0, 5 ),  0 );
    EXPECT_EQ( m.getElement( 0, 6 ),  0 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 22 );
    EXPECT_EQ( m.getElement( 1, 1 ), 22 );
    EXPECT_EQ( m.getElement( 1, 2 ), 22 );
    EXPECT_EQ( m.getElement( 1, 3 ),  0 );
    EXPECT_EQ( m.getElement( 1, 4 ),  0 );
    EXPECT_EQ( m.getElement( 1, 5 ),  0 );
    EXPECT_EQ( m.getElement( 1, 6 ),  0 );
    
    EXPECT_EQ( m.getElement( 2, 0 ),  0 );
    EXPECT_EQ( m.getElement( 2, 1 ),  0 );
    EXPECT_EQ( m.getElement( 2, 2 ),  0 );
    EXPECT_EQ( m.getElement( 2, 3 ), 33 );
    EXPECT_EQ( m.getElement( 2, 4 ), 33 );
    EXPECT_EQ( m.getElement( 2, 5 ), 33 );
    EXPECT_EQ( m.getElement( 2, 6 ),  0 );
}

template< typename Matrix >
void test_VectorProduct()
{
/*
 * Sets up the following 5x4 sparse matrix:
 *
 *    /  1  2  3  0 \
 *    |  0  0  0  4 |
 *    |  5  6  7  0 |
 *    |  0  8  9 10 |
 *    \  0  0 11 12 /
 */
    const int m_rows = 5;
    const int m_cols = 4;
    
    Matrix m;
    m.reset();
    m.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
    m.setCompressedRowLengths( rowLengths );
    
    int value = 1;
    for( int i = 0; i < m_cols - 1; i++ )   // 0th row
        m.setElement( 0, i, value++ );
    
        m.setElement( 1, 3, value++ );      // 1st row
        
    for( int i = 0; i < m_cols - 1; i++ )   // 2nd row
        m.setElement( 2, i, value++ );
        
    for( int i = 1; i < m_cols; i++ )       // 3rd row
        m.setElement( 3, i, value++ );
        
    for( int i = 2; i < m_cols; i++ )       // 4th row
        m.setElement( 4, i, value++ );
    
    #include <TNL/Containers/Vector.h>
    #include <TNL/Containers/VectorView.h>

    using namespace TNL;
    using namespace TNL::Containers;
    using namespace TNL::Containers::Algorithms;
    
    typedef typename Matrix::RealType RealType;
    typedef typename Matrix::DeviceType DeviceType;
    typedef typename Matrix::IndexType IndexType;

    Vector< RealType, DeviceType, IndexType > inVector;
    inVector.setSize( 4 );
    for( int i = 0; i < inVector.getSize(); i++ )        
        inVector.setElement( i, 2 );

    Vector< RealType, DeviceType, IndexType > outVector;  
    outVector.setSize( 5 );
    for( int j = 0; j < outVector.getSize(); j++ )
        outVector.setElement( j, 0 );
 
    
    m.vectorProduct( inVector, outVector);
   
    EXPECT_EQ( outVector.getElement( 0 ), 12 );
    EXPECT_EQ( outVector.getElement( 1 ),  8 );
    EXPECT_EQ( outVector.getElement( 2 ), 36 );
    EXPECT_EQ( outVector.getElement( 3 ), 54 );
    EXPECT_EQ( outVector.getElement( 4 ), 46 );
}

template< typename Matrix >
void test_PerformSORIteration()
{
/*
 * Sets up the following 4x4 sparse matrix:
 *
 *    /  4  1  0  0 \
 *    |  1  4  1  0 |
 *    |  0  1  4  1 |
 *    \  0  0  1  4 /
 */
    const int m_rows = 4;
    const int m_cols = 4;
    
    Matrix m;
    m.reset();
    m.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
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
    
    typedef typename Matrix::RealType RealType;
    typedef typename Matrix::DeviceType DeviceType;
    typedef typename Matrix::IndexType IndexType;
    
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
void test_SaveAndLoad()
{
/*
 * Sets up the following 4x4 sparse matrix:
 *
 *    /  1  2  3  0 \
 *    |  0  4  0  5 |
 *    |  6  7  8  0 |
 *    \  0  9 10 11 /
 */
    const int m_rows = 4;
    const int m_cols = 4;
    
    Matrix savedMatrix;
    savedMatrix.reset();
    savedMatrix.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
    savedMatrix.setCompressedRowLengths( rowLengths );
    
    int value = 1;
    for( int i = 0; i < m_cols - 1; i++ )   // 0th row
        savedMatrix.setElement( 0, i, value++ );
        
        savedMatrix.setElement( 1, 1, value++ );
        savedMatrix.setElement( 1, 3, value++ );      // 1st row
        
    for( int i = 0; i < m_cols - 1; i++ )   // 2nd row
        savedMatrix.setElement( 2, i, value++ );
        
    for( int i = 1; i < m_cols; i++ )       // 3rd row
        savedMatrix.setElement( 3, i, value++ );
        
    savedMatrix.save( "sparseMatrixFile" );
    
    Matrix loadedMatrix;
    loadedMatrix.reset();
    loadedMatrix.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths2;
    rowLengths2.setSize( m_rows );
    rowLengths2.setValue( 3 );
    loadedMatrix.setCompressedRowLengths( rowLengths2 );
    
    
    loadedMatrix.load( "sparseMatrixFile" );
    
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
    
    std::cout << "\nThis will create a file called 'sparseMatrixFile' (of the matrix created in the test function), in .../tnl-dev/Debug/bin/\n\n";
}

template< typename Matrix >
void test_Print()
{
/*
 * Sets up the following 5x4 sparse matrix:
 *
 *    /  1  2  3  0 \
 *    |  0  0  0  4 |
 *    |  5  6  7  0 |
 *    |  0  8  9 10 |
 *    \  0  0 11 12 /
 */
    const int m_rows = 5;
    const int m_cols = 4;
    
    Matrix m;
    m.reset();
    m.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
    m.setCompressedRowLengths( rowLengths );
    
    int value = 1;
    for( int i = 0; i < m_cols - 1; i++ )   // 0th row
        m.setElement( 0, i, value++ );
    
        m.setElement( 1, 3, value++ );      // 1st row
        
    for( int i = 0; i < m_cols - 1; i++ )   // 2nd row
        m.setElement( 2, i, value++ );
        
    for( int i = 1; i < m_cols; i++ )       // 3rd row
        m.setElement( 3, i, value++ );
        
    for( int i = 2; i < m_cols; i++ )       // 4th row
        m.setElement( 4, i, value++ );
    
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
    couted << "Row: 0 ->  Col:0->1	 Col:1->2	 Col:2->3\t\n"
               "Row: 1 ->  Col:3->4\t\n"
               "Row: 2 ->  Col:0->5	 Col:1->6	 Col:2->7\t\n"
               "Row: 3 ->  Col:1->8	 Col:2->9	 Col:3->10\t\n"
               "Row: 4 ->  Col:2->11	 Col:3->12\t\n";
    
    EXPECT_EQ( printed.str(), couted.str() );
}

//// test_getType is not general enough yet. DO NOT TEST IT YET.

//TEST( SparseMatrixTest, CSR_GetTypeTest_Host )
//{
//    host_test_GetType< CSR_host_float, CSR_host_int >();
//}
//
//#ifdef HAVE_CUDA
//TEST( SparseMatrixTest, CSR_GetTypeTest_Cuda )
//{
//    cuda_test_GetType< CSR_cuda_float, CSR_cuda_int >();
//}
//#endif

TEST( SparseMatrixTest, CSR_setDimensionsTest_Host )
{
    test_SetDimensions< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_setDimensionsTest_Cuda )
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

TEST( SparseMatrixTest, CSR_addElementTest_Host )
{
    test_AddElement< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_addElementTest_Cuda )
{
    test_AddElement< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_setRowTest_Host )
{
    test_SetRow< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_setRowTest_Cuda )
{
    test_SetRow< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_vectorProductTest_Host )
{
    test_VectorProduct< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_vectorProductTest_Cuda )
{
    test_VectorProduct< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_perforSORIterationTest_Host )
{
    test_PerformSORIteration< CSR_host_float >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_perforSORIterationTest_Cuda )
{
//    test_PerformSORIteration< CSR_cuda_float >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      [1]    16958 segmentation fault (core dumped)  ./SparseMatrixTest-dbg\n\n";
}
#endif

TEST( SparseMatrixTest, CSR_saveAndLoadTest_Host )
{
    test_SaveAndLoad< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_saveAndLoadTest_Cuda )
{
    test_SaveAndLoad< CSR_cuda_int >();
}
#endif

TEST( SparseMatrixTest, CSR_printTest_Host )
{
    test_Print< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_printTest_Cuda )
{
    test_Print< CSR_cuda_int >();
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

