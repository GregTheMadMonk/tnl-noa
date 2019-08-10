/***************************************************************************
                          SparseMatrixTest_ChunkedEllpack.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/ChunkedEllpack.h>

#include "SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class ChunkedEllpackMatrixTest : public ::testing::Test
{
protected:
   using ChunkedEllpackMatrixType = Matrix;
};

// columnIndexes of ChunkedEllpack appear to be broken, when printed, it prints out a bunch of 4s.
// rowPointers have interesting elements? 0 18 36 42 54 72 96 126 162 204 256 when rows = 10, cols = 11; rowLengths = 3 3 1 2 3 4 5 6 7 8
// and 0 52 103 154 205 256 when rows = 5, cols = 4; rowLengths = 3 3 3 3 3


// types for which MatrixTest is instantiated
using ChEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, short >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, short >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, short >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, short >,
    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
    ,TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ChunkedEllpackMatrixTest, ChEllpackMatrixTypes);

TYPED_TEST( ChunkedEllpackMatrixTest, setDimensionsTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetDimensions< ChunkedEllpackMatrixType >();
}

//TYPED_TEST( ChunkedEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
//    
////    test_SetCompressedRowLengths< ChunkedEllpackMatrixType >();
//    
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( ChunkedEllpackMatrixTest, setLikeTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetLike< ChunkedEllpackMatrixType, ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, resetTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Reset< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, addElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_AddElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setRowTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetRow< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, vectorProductTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_VectorProduct< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, saveAndLoadTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SaveAndLoad< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, printTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Print< ChunkedEllpackMatrixType >();
}

#endif

#include "../main.h"
