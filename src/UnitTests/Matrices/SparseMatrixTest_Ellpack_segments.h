/***************************************************************************
                          SparseMatrixTest_Ellpack_segments.h -  description
                             -------------------
    begin                : Dec 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class EllpackMatrixTest : public ::testing::Test
{
protected:
   using EllpackMatrixType = Matrix;
};

////
// Row-major format is used for the host system
template< typename Device, typename Index, typename IndexAlocator >
using RowMajorEllpack = TNL::Containers::Segments::Ellpack< Device, Index, IndexAlocator, true, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorEllpack = TNL::Containers::Segments::Ellpack< Device, Index, IndexAllocator, false, 32 >;

// types for which MatrixTest is instantiated
using EllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >
#endif
>;

TYPED_TEST_SUITE( EllpackMatrixTest, EllpackMatrixTypes);

TYPED_TEST( EllpackMatrixTest, setDimensionsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetDimensions< EllpackMatrixType >();
}

//TYPED_TEST( EllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
//
////    test_SetCompressedRowLengths< EllpackMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( EllpackMatrixTest, setLikeTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetLike< EllpackMatrixType, EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, resetTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Reset< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, getRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_GetRow< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, addElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_AddElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_VectorProduct< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, rowsReduction )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_RowsReduction< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, saveAndLoadTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SaveAndLoad< EllpackMatrixType >( "test_SparseMatrixTest_Ellpack_segments" );
}

TYPED_TEST( EllpackMatrixTest, printTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Print< EllpackMatrixType >();
}

#endif

#include "../main.h"
