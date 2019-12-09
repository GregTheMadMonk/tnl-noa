/***************************************************************************
                          SparseMatrixTest_SlicedEllpack_segments.h -  description
                             -------------------
    begin                : Dec 9, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class SlicedEllpackMatrixTest : public ::testing::Test
{
protected:
   using SlicedEllpackMatrixType = Matrix;
};

////
// Row-major format is used for the host system
template< typename Device, typename Index >
using RowMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, true, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index >
using ColumnMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, false, 32 >;

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     RowMajorSlicedEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< long,    RowMajorSlicedEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< float,   RowMajorSlicedEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< double,  RowMajorSlicedEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< int,     RowMajorSlicedEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< long,    RowMajorSlicedEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< float,   RowMajorSlicedEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< double,  RowMajorSlicedEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< int,     RowMajorSlicedEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< long,    RowMajorSlicedEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< float,   RowMajorSlicedEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< double,  RowMajorSlicedEllpack, TNL::Devices::Host, long  >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     ColumnMajorSlicedEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorSlicedEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorSlicedEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorSlicedEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< int,     ColumnMajorSlicedEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorSlicedEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorSlicedEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorSlicedEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< int,     ColumnMajorSlicedEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorSlicedEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorSlicedEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorSlicedEllpack, TNL::Devices::Cuda, long  >
#endif
>;

TYPED_TEST_SUITE( SlicedEllpackMatrixTest, SlicedEllpackMatrixTypes);

TYPED_TEST( SlicedEllpackMatrixTest, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetDimensions< SlicedEllpackMatrixType >();
}

//TYPED_TEST( SlicedEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
//
////    test_SetCompressedRowLengths< SlicedEllpackMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( SlicedEllpackMatrixTest, setLikeTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetLike< SlicedEllpackMatrixType, SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, resetTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Reset< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, addElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_AddElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetRow< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, saveAndLoadTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SaveAndLoad< SlicedEllpackMatrixType >( "test_SparseMatrixTest_SlicedEllpack_segments" );
}

TYPED_TEST( SlicedEllpackMatrixTest, printTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Print< SlicedEllpackMatrixType >();
}

#endif

#include "../main.h"
