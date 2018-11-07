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
 * getTypeVirtual()                 ::TEST? This just calls getType().
 * getSerializationType()           ::TEST? This just calls HostType::getType().
 * getSerializationTypeVirtual()    ::TEST? This just calls getSerializationType().
 * setDimensions()                      ::DONE
 * setCompressedRowLengths()            ::DONE
 * getRowLength()                   ::USED! In test_SetCompressedRowLengths() to verify the test itself.
 * getRowLengthFast()               ::TEST? How to test __cuda_callable__?
 * setLike()                            ::DONE
 * reset()                              ::DONE
 * setElementFast()                 ::TEST? How to test __cuda_callable__?
 * setElement()                         ::DONE
 * addElementFast()                 ::TEST? How to test __cuda_callable__?
 * addElement()                     ::HOW?  How to use the thisElementMultiplicator? Does it need testing?
 * setRowFast()                     ::TEST? How to test __cuda_callable__?
 * setRow()                             ::DONE
 * addRowFast()                     ::TEST? How to test __cuda_callable__?
 * addRow()                         ::NOT IMPLEMENTED! Implement? Is it supposed to add an extra row to the matrix or arr elements of a row to another row in the matrix?
 * getElementFast()                 ::TEST? How to test __cuda_callable__?
 * getElement()                     ::USED! In test_SetElement(), test_AddElement() and test_setRow() to verify the test itself.
 * getRowFast()                     ::TEST? How to test __cuda_callable__?
 * MatrixRow getRow()               ::TEST? How to test __cuda_callable__?
 * ConstMatrixRow getRow()          ::TEST? How to test __cuda_callable__?
 * rowVectorProduct()               ::TEST? How to test __cuda_callable__?
 * vectorProduct()                  ::HOW? Throwing errors in CSR_impl.h (779) no instance matches the arguments when using int arrays. When tried using Vector_impl.h index out of bounds or CUDA illegal memory access
 * addMatrix()                      ::NOT IMPLEMENTED!
 * getTransposition()               ::NOT IMPLMENETED!
 * performSORIteration()            ::HOW? What does this do? Ax=b but splitting A into D(:=Diagonal) L(:=Lower Triangular) U(=Upper Triangular) matrices. What is the omega(relaxation/residual factor??)? https://en.wikipedia.org/wiki/Successive_over-relaxation
 * operator=()                      ::HOW? What is this supposed to enable? Overloading operators?
 * save( File& file)                ::USED! In save( String& fileName )
 * load( File& file )               ::USED! In load( String& fileName )
 * save( String& fileName )             ::DONE
 * load( String& fileName )             ::DONE
 * print()
 * setCudaKernelType()
 * getCudaKernelType()              ::TEST? How to test __cuda_callable__?
 * setCudaWarpSize()
 * getCudaWarpSize()
 * setHybridModeSplit()
 * getHybridModeSplit()             ::TEST? How to test __cuda_callable__?
 * spmvCudaVectorized()             ::TEST? How to test __device__?
 * vectorProductCuda()              ::TEST? How to test __device__?
 */

// GENERAL TODO
/*
 * For every function, EXPECT_EQ needs to be done, even for zeros in matrices.
 */


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
    for( int i = 0; i < rows; i++ )
        m.addElement( i, 0, value++, 0.0 );
    
    m.addElement( 0, 4, 1, 0.0 );
    
    EXPECT_EQ( m.getElement( 0, 0 ), 1 );
    EXPECT_EQ( m.getElement( 0, 1 ), 0 );
    EXPECT_EQ( m.getElement( 0, 2 ), 0 );
    EXPECT_EQ( m.getElement( 0, 3 ), 0 );
    EXPECT_EQ( m.getElement( 0, 4 ), 1 );
    
    EXPECT_EQ( m.getElement( 1, 0 ), 2 );
    EXPECT_EQ( m.getElement( 1, 1 ), 0 );
    EXPECT_EQ( m.getElement( 1, 2 ), 0 );
    EXPECT_EQ( m.getElement( 1, 3 ), 0 );
    EXPECT_EQ( m.getElement( 1, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 2, 0 ), 3 );
    EXPECT_EQ( m.getElement( 2, 1 ), 0 );
    EXPECT_EQ( m.getElement( 2, 2 ), 0 );
    EXPECT_EQ( m.getElement( 2, 3 ), 0 );
    EXPECT_EQ( m.getElement( 2, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 3, 0 ), 4 );
    EXPECT_EQ( m.getElement( 3, 1 ), 0 );
    EXPECT_EQ( m.getElement( 3, 2 ), 0 );
    EXPECT_EQ( m.getElement( 3, 3 ), 0 );
    EXPECT_EQ( m.getElement( 3, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 4, 0 ), 5 );
    EXPECT_EQ( m.getElement( 4, 1 ), 0 );
    EXPECT_EQ( m.getElement( 4, 2 ), 0 );
    EXPECT_EQ( m.getElement( 4, 3 ), 0 );
    EXPECT_EQ( m.getElement( 4, 4 ), 0 );
    
    EXPECT_EQ( m.getElement( 5, 0 ), 6 );
    EXPECT_EQ( m.getElement( 5, 1 ), 0 );
    EXPECT_EQ( m.getElement( 5, 2 ), 0 );
    EXPECT_EQ( m.getElement( 5, 3 ), 0 );
    EXPECT_EQ( m.getElement( 5, 4 ), 0 );
}

template< typename Matrix >
void test_SetRow()
{
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
    
    int row1 [ 3 ] = { 11, 11, 11 }; int colIndexes1 [3] = { 0, 1, 2 };
    int row2 [ 3 ] = { 22, 22, 22 }; int colIndexes2 [3] = { 0, 1, 2 };
    int row3 [ 3 ] = { 33, 33, 33 }; int colIndexes3 [3] = { 3, 4, 5 };
    
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
    bool testRan = false;
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
    
//                        #include <TNL/Containers/Vector.h>
//                        #include <TNL/Containers/VectorView.h>
//                    
//                        using namespace TNL;
//                        using namespace TNL::Containers;
//                        using namespace TNL::Containers::Algorithms;
//                        
//                        Vector< int, Devices::Host, int > inVector;
//                        inVector.setSize( 5 );
//                        for( int i = 0; i < inVector.getSize(); i++ )        
//                            inVector.setElement( i, 1 );
//                        
//                        Vector< int, Devices::Host, int > outVector;  
//                        outVector.setSize( 4 );                        // ERROR: out of bounds, if set to 3 or 4. CUDA illegal memory access when set to 5.
//                        for( int j = 0; j < outVector.getSize(); j++ )
//                            outVector.setElement( j, 0 );//outVector[ j ] = 0;
    
//    const int inVector [ 5 ] = { 1, 1, 1, 1, 1 };
//    int outVector [ 4 ] = { 0, 0, 0, 0 };
//    
//    m.vectorProduct( inVector, outVector); // ERROR: This throws an error when Vector<> declarations are used.
//    testRan = true;    
//    EXPECT_EQ( outVector.getElement( 0 ),  6 );
//    EXPECT_EQ( outVector.getElement( 1 ), 16 );
//    EXPECT_EQ( outVector.getElement( 2 ), 30 );
//    EXPECT_EQ( outVector.getElement( 3 ), 26 );
        
    EXPECT_TRUE( testRan );
    std::cout << "TEST DID NOT RUN. NOT IMPLETENTED.\n";
}

template< typename Matrix >
void test_PerformSORIteration()
{
/*
 * Sets up the following 5x4 sparse matrix:
 *
 *    /  1  2  3  0 \
 *    |  0  4  0  5 |
 *    |  6  7  8  0 |
 *    \  0  9 10 11 /
 */
    bool testRan = false;
//    const int m_rows = 4;
//    const int m_cols = 4;
//    
//    Matrix m;
//    m.reset();
//    m.setDimensions( m_rows, m_cols );
//    typename Matrix::CompressedRowLengthsVector rowLengths;
//    rowLengths.setSize( m_rows );
//    rowLengths.setValue( 3 );
//    m.setCompressedRowLengths( rowLengths );
//    
//    int value = 1;
//    for( int i = 0; i < m_cols - 1; i++ )   // 0th row
//        m.setElement( 0, i, value++ );
//        
//        m.setElement( 1, 1, value++ );
//        m.setElement( 1, 3, value++ );      // 1st row
//        
//    for( int i = 0; i < m_cols - 1; i++ )   // 2nd row
//        m.setElement( 2, i, value++ );
//        
//    for( int i = 1; i < m_cols; i++ )       // 3rd row
//        m.setElement( 3, i, value++ );
//    
//    // Print out the Matrix
//    std::cout << "Matrix m: \n";
//    for( int i = 0; i < m_rows; i++ )
//    {
//        std::cout << "| ";
//        for(int j = 0; j < m_cols; j++ )
//            std::cout << m.getElement( i, j ) << " ";
//        std::cout << " |\n";
//    }
//    std::cout << std::endl;
//    
//    int bVector [ 4 ] = { 6, 9, 21, 30 };
//    int xVector [ 4 ] = { 1, 1, 1, 1 };
//    
//    m.performSORIteration( bVector, 0, xVector, 1);
//    m.performSORIteration( bVector, 1, xVector, 1);
//    m.performSORIteration( bVector, 2, xVector, 1);
//    m.performSORIteration( bVector, 3, xVector, 1);
//    
//    std::cout << "\n[ ";
//    for( int i = 0; i < 4; i++ )
//        std::cout << xVector[ i ] << " ";
//    std::cout << " ]\n";
//    
//    std::cout << "\n[ ";
//    for( int i = 0; i < 4; i++ )
//        std::cout << bVector[ i ] << " ";
//    std::cout << " ]\n";
//    
//    testRan = true;
//    EXPECT_EQ( xVector[ 0 ], 1 );
//    EXPECT_EQ( xVector[ 1 ], 1 );
//    EXPECT_EQ( xVector[ 2 ], 1 );
//    EXPECT_EQ( xVector[ 3 ], 1 );
    
    EXPECT_TRUE( testRan );
    std::cout << "TEST DID NOT RUN. NOT IMPLETENTED.\n";
}

template< typename Matrix >
void test_SaveAndLoad()
{
/*
 * Sets up the following 5x4 sparse matrix:
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
        
    savedMatrix.save( "/home/lukas/m" );
    
    Matrix loadedMatrix;
    loadedMatrix.reset();
    loadedMatrix.setDimensions( m_rows, m_cols );
    typename Matrix::CompressedRowLengthsVector rowLengths2;
    rowLengths2.setSize( m_rows );
    rowLengths2.setValue( 3 );
    loadedMatrix.setCompressedRowLengths( rowLengths2 );
    
    
    loadedMatrix.load( "/home/lukas/m" );
    
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
    test_PerformSORIteration< CSR_host_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_perforSORIterationTest_Cuda )
{
    test_PerformSORIteration< CSR_cuda_int >();
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

