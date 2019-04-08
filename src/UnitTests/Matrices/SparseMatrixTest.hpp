/***************************************************************************
                          SparseMatrixTest_impl.h -  description
                             -------------------
    begin                : Nov 22, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

// Temporary, until test_OperatorEquals doesn't work for all formats.
#include <TNL/Matrices/ChunkedEllpack.h>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

template< typename MatrixHostFloat, typename MatrixHostInt >
void host_test_GetType()
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
    
//    MatrixHostFloat mtrxHostFloat;
//    MatrixHostInt mtrxHostInt;
//    
//    EXPECT_EQ( mtrxHostFloat.getType(), TNL::String( "Matrices::CSR< float, Devices::Host >" ) );
//    EXPECT_EQ( mtrxHostInt.getType(), TNL::String( "Matrices::CSR< int, Devices::Host >" ) ); 
}

template< typename MatrixCudaFloat, typename MatrixCudaInt >
void cuda_test_GetType()
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
    
//    MatrixCudaFloat mtrxCudaFloat;
//    MatrixCudaInt mtrxCudaInt;
//    
//    EXPECT_EQ( mtrxCudaFloat.getType(), TNL::String( "Matrices::CSR< float, Devices::Cuda >" ) );
//    EXPECT_EQ( mtrxCudaInt.getType(), TNL::String( "Matrices::CSR< int, Devices::Cuda >" ) );        
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
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 3 );
    
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
    
    
    EXPECT_EQ( m.getNonZeroRowLength( 0 ), 3 );
    EXPECT_EQ( m.getNonZeroRowLength( 1 ), 3 );
    EXPECT_EQ( m.getNonZeroRowLength( 2 ), 1 );
    EXPECT_EQ( m.getNonZeroRowLength( 3 ), 2 );
    EXPECT_EQ( m.getNonZeroRowLength( 4 ), 3 );
    EXPECT_EQ( m.getNonZeroRowLength( 5 ), 4 );
    EXPECT_EQ( m.getNonZeroRowLength( 6 ), 5 );
    EXPECT_EQ( m.getNonZeroRowLength( 7 ), 6 );
    EXPECT_EQ( m.getNonZeroRowLength( 8 ), 7 );
    EXPECT_EQ( m.getNonZeroRowLength( 9 ), 8 );
    
//    if( m.getType() == TNL::String( TNL::String( "Matrices::CSR< ") +
//                       TNL::String( TNL::getType< RealType >() ) +
//                       TNL::String( ", " ) +
//                       TNL::String( Matrix::DeviceType::getDeviceType() ) +
//                       //TNL::String( ", " ) +
//                       //TNL::String( TNL::getType< IndexType >() ) +
//                       TNL::String( " >" ) )
//      )
//    {
//        EXPECT_EQ( m.getRowLength( 0 ), 3 );
//        EXPECT_EQ( m.getRowLength( 1 ), 3 );
//        EXPECT_EQ( m.getRowLength( 2 ), 1 );
//        EXPECT_EQ( m.getRowLength( 3 ), 2 );
//        EXPECT_EQ( m.getRowLength( 4 ), 3 );
//        EXPECT_EQ( m.getRowLength( 5 ), 4 );
//        EXPECT_EQ( m.getRowLength( 6 ), 5 );
//        EXPECT_EQ( m.getRowLength( 7 ), 6 );
//        EXPECT_EQ( m.getRowLength( 8 ), 7 );
//        EXPECT_EQ( m.getRowLength( 9 ), 8 );
//    }
//    else if( m.getType() == TNL::String( TNL::String( "Matrices::AdEllpack< ") +
//                            TNL::String( TNL::getType< RealType >() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( Matrix::DeviceType::getDeviceType() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( TNL::getType< IndexType >() ) +
//                            TNL::String( " >" ) ) 
//                            || 
//             m.getType() == TNL::String( TNL::String( "Matrices::SlicedEllpack< ") +
//                            TNL::String( TNL::getType< RealType >() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( Matrix::DeviceType::getDeviceType() ) +
//                            TNL::String( " >" ) )
//           )
//    {
//        EXPECT_EQ( m.getRowLength( 0 ), 8 );
//        EXPECT_EQ( m.getRowLength( 1 ), 8 );
//        EXPECT_EQ( m.getRowLength( 2 ), 8 );
//        EXPECT_EQ( m.getRowLength( 3 ), 8 );
//        EXPECT_EQ( m.getRowLength( 4 ), 8 );
//        EXPECT_EQ( m.getRowLength( 5 ), 8 );
//        EXPECT_EQ( m.getRowLength( 6 ), 8 );
//        EXPECT_EQ( m.getRowLength( 7 ), 8 );
//        EXPECT_EQ( m.getRowLength( 8 ), 8 );
//        EXPECT_EQ( m.getRowLength( 9 ), 8 );
//    }
//    else if( m.getType() == TNL::String( TNL::String( "Matrices::Ellpack< ") +
//                            TNL::String( TNL::getType< RealType >() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( Matrix::DeviceType::getDeviceType() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( TNL::getType< IndexType >() ) +
//                            TNL::String( " >" ) ) 
//                            ||
//             m.getType() == TNL::String( TNL::String( "Matrices::ChunkedEllpack< ") +
//                            TNL::String( TNL::getType< RealType >() ) +
//                            TNL::String( ", " ) +
//                            TNL::String( Matrix::DeviceType::getDeviceType() ) +
//                            TNL::String( " >" ) )
//           )
//    {
//        EXPECT_EQ( m.getNonZeroRowLength( 0 ), 3 );
//        EXPECT_EQ( m.getNonZeroRowLength( 1 ), 3 );
//        EXPECT_EQ( m.getNonZeroRowLength( 2 ), 1 );
//        EXPECT_EQ( m.getNonZeroRowLength( 3 ), 2 );
//        EXPECT_EQ( m.getNonZeroRowLength( 4 ), 3 );
//        EXPECT_EQ( m.getNonZeroRowLength( 5 ), 4 );
//        EXPECT_EQ( m.getNonZeroRowLength( 6 ), 5 );
//        EXPECT_EQ( m.getNonZeroRowLength( 7 ), 6 );
//        EXPECT_EQ( m.getNonZeroRowLength( 8 ), 7 );
//        EXPECT_EQ( m.getNonZeroRowLength( 9 ), 8 );
//    }
//    else
//    {
//        EXPECT_EQ( m.getRowLength( 0 ), 3 );
//        EXPECT_EQ( m.getRowLength( 1 ), 3 );
//        EXPECT_EQ( m.getRowLength( 2 ), 1 );
//        EXPECT_EQ( m.getRowLength( 3 ), 2 );
//        EXPECT_EQ( m.getRowLength( 4 ), 3 );
//        EXPECT_EQ( m.getRowLength( 5 ), 4 );
//        EXPECT_EQ( m.getRowLength( 6 ), 5 );
//        EXPECT_EQ( m.getRowLength( 7 ), 6 );
//        EXPECT_EQ( m.getRowLength( 8 ), 7 );
//        EXPECT_EQ( m.getRowLength( 9 ), 8 );
//    }
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
    
    Matrix m;
    m.setDimensions( rows, cols );
    
    m.reset();
    
    
    EXPECT_EQ( m.getRows(), 0 );
    EXPECT_EQ( m.getColumns(), 0 );
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
 *    /  1  0  0  0  0  0  0  0  0  0  \
 *    |  0  2  0  0  0  0  0  0  0  0  |
 *    |  0  0  3  0  0  0  0  0  0  0  |
 *    |  0  0  0  4  0  0  0  0  0  0  |
 *    |  0  0  0  0  5  0  0  0  0  0  |
 *    |  0  0  0  0  0  6  0  0  0  0  |
 *    |  0  0  0  0  0  0  7  0  0  0  |
 *    |  0  0  0  0  0  0  0  8  0  0  |
 *    |  0  0  0  0  0  0  0  0  9  0  |
 *    \  0  0  0  0  0  0  0  0  0 10  /
 */
    
    const IndexType rows = 10;
    const IndexType cols = 10;
    
    Matrix m;
    m.reset();
    
//    std::cout << "Test:\n\tMatrix reset." << std::endl;
    
    m.setDimensions( rows, cols );
    
//    std::cout << "\tMatrix dimensions set." << std::endl;
    
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    
//    std::cout << "\tRow lengths size set." << std::endl;
    
    rowLengths.setValue( 1 );
    
//    std::cout << "\tRow lengths value set." << std::endl;
    
    m.setCompressedRowLengths( rowLengths );
    
//    std::cout << "\tCompressed row lengths set." << std::endl;
    
    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        m.setElement( i, i, value++ );
    
    
    EXPECT_EQ( m.getElement( 0, 0 ),  1 );
    EXPECT_EQ( m.getElement( 0, 1 ),  0 );
    EXPECT_EQ( m.getElement( 0, 2 ),  0 );
    EXPECT_EQ( m.getElement( 0, 3 ),  0 );
    EXPECT_EQ( m.getElement( 0, 4 ),  0 );
    EXPECT_EQ( m.getElement( 0, 5 ),  0 );
    EXPECT_EQ( m.getElement( 0, 6 ),  0 );
    EXPECT_EQ( m.getElement( 0, 7 ),  0 );
    EXPECT_EQ( m.getElement( 0, 8 ),  0 );
    EXPECT_EQ( m.getElement( 0, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 1, 0 ),  0 );
    EXPECT_EQ( m.getElement( 1, 1 ),  2 );
    EXPECT_EQ( m.getElement( 1, 2 ),  0 );
    EXPECT_EQ( m.getElement( 1, 3 ),  0 );
    EXPECT_EQ( m.getElement( 1, 4 ),  0 );
    EXPECT_EQ( m.getElement( 1, 5 ),  0 );
    EXPECT_EQ( m.getElement( 1, 6 ),  0 );
    EXPECT_EQ( m.getElement( 1, 7 ),  0 );
    EXPECT_EQ( m.getElement( 1, 8 ),  0 );
    EXPECT_EQ( m.getElement( 1, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 2, 0 ),  0 );
    EXPECT_EQ( m.getElement( 2, 1 ),  0 );
    EXPECT_EQ( m.getElement( 2, 2 ),  3 );
    EXPECT_EQ( m.getElement( 2, 3 ),  0 );
    EXPECT_EQ( m.getElement( 2, 4 ),  0 );
    EXPECT_EQ( m.getElement( 2, 5 ),  0 );
    EXPECT_EQ( m.getElement( 2, 6 ),  0 );
    EXPECT_EQ( m.getElement( 2, 7 ),  0 );
    EXPECT_EQ( m.getElement( 2, 8 ),  0 );
    EXPECT_EQ( m.getElement( 2, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 3, 0 ),  0 );
    EXPECT_EQ( m.getElement( 3, 1 ),  0 );
    EXPECT_EQ( m.getElement( 3, 2 ),  0 );
    EXPECT_EQ( m.getElement( 3, 3 ),  4 );
    EXPECT_EQ( m.getElement( 3, 4 ),  0 );
    EXPECT_EQ( m.getElement( 3, 5 ),  0 );
    EXPECT_EQ( m.getElement( 3, 6 ),  0 );
    EXPECT_EQ( m.getElement( 3, 7 ),  0 );
    EXPECT_EQ( m.getElement( 3, 8 ),  0 );
    EXPECT_EQ( m.getElement( 3, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 4, 0 ),  0 );
    EXPECT_EQ( m.getElement( 4, 1 ),  0 );
    EXPECT_EQ( m.getElement( 4, 2 ),  0 );
    EXPECT_EQ( m.getElement( 4, 3 ),  0 );
    EXPECT_EQ( m.getElement( 4, 4 ),  5 );
    EXPECT_EQ( m.getElement( 4, 5 ),  0 );
    EXPECT_EQ( m.getElement( 4, 6 ),  0 );
    EXPECT_EQ( m.getElement( 4, 7 ),  0 );
    EXPECT_EQ( m.getElement( 4, 8 ),  0 );
    EXPECT_EQ( m.getElement( 4, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 5, 0 ),  0 );
    EXPECT_EQ( m.getElement( 5, 1 ),  0 );
    EXPECT_EQ( m.getElement( 5, 2 ),  0 );
    EXPECT_EQ( m.getElement( 5, 3 ),  0 );
    EXPECT_EQ( m.getElement( 5, 4 ),  0 );
    EXPECT_EQ( m.getElement( 5, 5 ),  6 );
    EXPECT_EQ( m.getElement( 5, 6 ),  0 );
    EXPECT_EQ( m.getElement( 5, 7 ),  0 );
    EXPECT_EQ( m.getElement( 5, 8 ),  0 );
    EXPECT_EQ( m.getElement( 5, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 6, 0 ),  0 );
    EXPECT_EQ( m.getElement( 6, 1 ),  0 );
    EXPECT_EQ( m.getElement( 6, 2 ),  0 );
    EXPECT_EQ( m.getElement( 6, 3 ),  0 );
    EXPECT_EQ( m.getElement( 6, 4 ),  0 );
    EXPECT_EQ( m.getElement( 6, 5 ),  0 );
    EXPECT_EQ( m.getElement( 6, 6 ),  7 );
    EXPECT_EQ( m.getElement( 6, 7 ),  0 );
    EXPECT_EQ( m.getElement( 6, 8 ),  0 );
    EXPECT_EQ( m.getElement( 6, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 7, 0 ),  0 );
    EXPECT_EQ( m.getElement( 7, 1 ),  0 );
    EXPECT_EQ( m.getElement( 7, 2 ),  0 );
    EXPECT_EQ( m.getElement( 7, 3 ),  0 );
    EXPECT_EQ( m.getElement( 7, 4 ),  0 );
    EXPECT_EQ( m.getElement( 7, 5 ),  0 );
    EXPECT_EQ( m.getElement( 7, 6 ),  0 );
    EXPECT_EQ( m.getElement( 7, 7 ),  8 );
    EXPECT_EQ( m.getElement( 7, 8 ),  0 );
    EXPECT_EQ( m.getElement( 7, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 8, 0 ),  0 );
    EXPECT_EQ( m.getElement( 8, 1 ),  0 );
    EXPECT_EQ( m.getElement( 8, 2 ),  0 );
    EXPECT_EQ( m.getElement( 8, 3 ),  0 );
    EXPECT_EQ( m.getElement( 8, 4 ),  0 );
    EXPECT_EQ( m.getElement( 8, 5 ),  0 );
    EXPECT_EQ( m.getElement( 8, 6 ),  0 );
    EXPECT_EQ( m.getElement( 8, 7 ),  0 );
    EXPECT_EQ( m.getElement( 8, 8 ),  9 );
    EXPECT_EQ( m.getElement( 8, 9 ),  0 );
    
    EXPECT_EQ( m.getElement( 9, 0 ),  0 );
    EXPECT_EQ( m.getElement( 9, 1 ),  0 );
    EXPECT_EQ( m.getElement( 9, 2 ),  0 );
    EXPECT_EQ( m.getElement( 9, 3 ),  0 );
    EXPECT_EQ( m.getElement( 9, 4 ),  0 );
    EXPECT_EQ( m.getElement( 9, 5 ),  0 );
    EXPECT_EQ( m.getElement( 9, 6 ),  0 );
    EXPECT_EQ( m.getElement( 9, 7 ),  0 );
    EXPECT_EQ( m.getElement( 9, 8 ),  0 );
    EXPECT_EQ( m.getElement( 9, 9 ), 10 );
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
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 3 );
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
void test_SetRow()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
    
/*
 * Sets up the following 3x7 sparse matrix:
 *
 *    /  0  0  0  1  1  1  0 \
 *    |  2  2  2  0  0  0  0 |
 *    \  3  3  3  0  0  0  0 /
 */
    
    const IndexType rows = 3;
    const IndexType cols = 7;
    
    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( rows );
    rowLengths.setValue( 6 );
    rowLengths.setElement( 1, 3 );
    m.setCompressedRowLengths( rowLengths );
    
    RealType value = 1;
    for( IndexType i = 0; i < 3; i++ )
    {
        m.setElement( 0, i + 3, value );
        m.setElement( 1, i, value + 1 );
        m.setElement( 2, i, value + 2 );
    }
    
    RealType row1 [ 3 ] = { 11, 11, 11 }; IndexType colIndexes1 [ 3 ] = { 0, 1, 2 };
    RealType row2 [ 3 ] = { 22, 22, 22 }; IndexType colIndexes2 [ 3 ] = { 0, 1, 2 };
    RealType row3 [ 3 ] = { 33, 33, 33 }; IndexType colIndexes3 [ 3 ] = { 3, 4, 5 };
    
    RealType row = 0;
    IndexType elements = 3;
    
    m.setRow( row++, colIndexes1, row1, elements );
    m.setRow( row++, colIndexes2, row2, elements );
    m.setRow( row++, colIndexes3, row3, elements );
    
    
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
    
    Matrix m;
    m.reset();
    m.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
    m.setCompressedRowLengths( rowLengths );
    
    RealType value = 1;
    for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
        m.setElement( 0, i, value++ );
    
    m.setElement( 1, 3, value++ );      // 1st row
        
    for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
        m.setElement( 2, i, value++ );
        
    for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
        m.setElement( 3, i, value++ );
        
    for( IndexType i = 2; i < m_cols; i++ )       // 4th row
        m.setElement( 4, i, value++ );

    using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    
    VectorType inVector;
    inVector.setSize( m_cols );
    for( IndexType i = 0; i < inVector.getSize(); i++ )        
        inVector.setElement( i, 2 );

    VectorType outVector;  
    outVector.setSize( m_rows );
    for( IndexType j = 0; j < outVector.getSize(); j++ )
        outVector.setElement( j, 0 );
 
    
    m.vectorProduct( inVector, outVector );
    
   
    EXPECT_EQ( outVector.getElement( 0 ), 12 );
    EXPECT_EQ( outVector.getElement( 1 ),  8 );
    EXPECT_EQ( outVector.getElement( 2 ), 36 );
    EXPECT_EQ( outVector.getElement( 3 ), 54 );
    EXPECT_EQ( outVector.getElement( 4 ), 46 );
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

// This test is only for Chunked Ellpack
template< typename Matrix >
void test_OperatorEquals()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   
   if( std::is_same< DeviceType, TNL::Devices::Cuda >::value )
       return;
   else
   {
       using CHELL_host = TNL::Matrices::ChunkedEllpack< RealType, TNL::Devices::Host, IndexType >;
       using CHELL_cuda = TNL::Matrices::ChunkedEllpack< RealType, TNL::Devices::Cuda, IndexType >;

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

        CHELL_host m_host;

        m_host.reset();
        m_host.setDimensions( m_rows, m_cols );
        typename CHELL_host::CompressedRowLengthsVector rowLengths;
        rowLengths.setSize( m_rows );
        rowLengths.setValue( 3 );
        m_host.setCompressedRowLengths( rowLengths );

        RealType value = 1;
        for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
            m_host.setElement( 0, i, value++ );

        m_host.setElement( 1, 1, value++ );
        m_host.setElement( 1, 3, value++ );           // 1st row

        for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
            m_host.setElement( 2, i, value++ );

        for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
            m_host.setElement( 3, i, value++ );

        EXPECT_EQ( m_host.getElement( 0, 0 ),  1 );
        EXPECT_EQ( m_host.getElement( 0, 1 ),  2 );
        EXPECT_EQ( m_host.getElement( 0, 2 ),  3 );
        EXPECT_EQ( m_host.getElement( 0, 3 ),  0 );

        EXPECT_EQ( m_host.getElement( 1, 0 ),  0 );
        EXPECT_EQ( m_host.getElement( 1, 1 ),  4 );
        EXPECT_EQ( m_host.getElement( 1, 2 ),  0 );
        EXPECT_EQ( m_host.getElement( 1, 3 ),  5 );

        EXPECT_EQ( m_host.getElement( 2, 0 ),  6 );
        EXPECT_EQ( m_host.getElement( 2, 1 ),  7 );
        EXPECT_EQ( m_host.getElement( 2, 2 ),  8 );
        EXPECT_EQ( m_host.getElement( 2, 3 ),  0 );

        EXPECT_EQ( m_host.getElement( 3, 0 ),  0 );
        EXPECT_EQ( m_host.getElement( 3, 1 ),  9 );
        EXPECT_EQ( m_host.getElement( 3, 2 ), 10 );
        EXPECT_EQ( m_host.getElement( 3, 3 ), 11 );

        CHELL_cuda m_cuda;

        // Copy the host matrix into the cuda matrix
        m_cuda = m_host;

        // Reset the host matrix
        m_host.reset();

        // Copy the cuda matrix back into the host matrix
        m_host = m_cuda;

        // Check the newly created double-copy host matrix
        EXPECT_EQ( m_host.getElement( 0, 0 ),  1 );
        EXPECT_EQ( m_host.getElement( 0, 1 ),  2 );
        EXPECT_EQ( m_host.getElement( 0, 2 ),  3 );
        EXPECT_EQ( m_host.getElement( 0, 3 ),  0 );

        EXPECT_EQ( m_host.getElement( 1, 0 ),  0 );
        EXPECT_EQ( m_host.getElement( 1, 1 ),  4 );
        EXPECT_EQ( m_host.getElement( 1, 2 ),  0 );
        EXPECT_EQ( m_host.getElement( 1, 3 ),  5 );

        EXPECT_EQ( m_host.getElement( 2, 0 ),  6 );
        EXPECT_EQ( m_host.getElement( 2, 1 ),  7 );
        EXPECT_EQ( m_host.getElement( 2, 2 ),  8 );
        EXPECT_EQ( m_host.getElement( 2, 3 ),  0 );

        EXPECT_EQ( m_host.getElement( 3, 0 ),  0 );
        EXPECT_EQ( m_host.getElement( 3, 1 ),  9 );
        EXPECT_EQ( m_host.getElement( 3, 2 ), 10 );
        EXPECT_EQ( m_host.getElement( 3, 3 ), 11 );
        
        // Try vectorProduct with copied cuda matrix to see if it works correctly.
        using VectorType = TNL::Containers::Vector< RealType, TNL::Devices::Cuda, IndexType >;
    
        VectorType inVector;
        inVector.setSize( m_cols );
        for( IndexType i = 0; i < inVector.getSize(); i++ )        
            inVector.setElement( i, 2 );

        VectorType outVector;  
        outVector.setSize( m_rows );
        for( IndexType j = 0; j < outVector.getSize(); j++ )
            outVector.setElement( j, 0 );
        
        m_cuda.vectorProduct( inVector, outVector );
        
        EXPECT_EQ( outVector.getElement( 0 ), 12 );
        EXPECT_EQ( outVector.getElement( 1 ), 18 );
        EXPECT_EQ( outVector.getElement( 2 ), 42 );
        EXPECT_EQ( outVector.getElement( 3 ), 60 );
   }
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
    
    Matrix savedMatrix;
    savedMatrix.reset();
    savedMatrix.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
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
    
    Matrix loadedMatrix;
    loadedMatrix.reset();
    loadedMatrix.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths2;
    rowLengths2.setSize( m_rows );
    rowLengths2.setValue( 3 );
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
    
    Matrix m;
    m.reset();
    m.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths;
    rowLengths.setSize( m_rows );
    rowLengths.setValue( 3 );
    m.setCompressedRowLengths( rowLengths );
    
    RealType value = 1;
    for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
        m.setElement( 0, i, value++ );
    
    m.setElement( 1, 3, value++ );      // 1st row
        
    for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
        m.setElement( 2, i, value++ );
        
    for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
        m.setElement( 3, i, value++ );
        
    for( IndexType i = 2; i < m_cols; i++ )       // 4th row
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

#endif
