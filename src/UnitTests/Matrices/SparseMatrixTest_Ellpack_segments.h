/***************************************************************************
                          SparseMatrixTest_Ellpack.h -  description
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
template< typename Device, typename Index >
using RowMajorEllpack = TNL::Containers::Segments::Ellpack< Device, Index, true, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index >
using ColumnMajorEllpack = TNL::Containers::Segments::Ellpack< Device, Index, false, 32 >;

// types for which MatrixTest is instantiated
using EllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     RowMajorEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< long,    RowMajorEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< float,   RowMajorEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< double,  RowMajorEllpack, TNL::Devices::Host, short >,
    TNL::Matrices::SparseMatrix< int,     RowMajorEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< long,    RowMajorEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< float,   RowMajorEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< double,  RowMajorEllpack, TNL::Devices::Host, int   >,
    TNL::Matrices::SparseMatrix< int,     RowMajorEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< long,    RowMajorEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< float,   RowMajorEllpack, TNL::Devices::Host, long  >,
    TNL::Matrices::SparseMatrix< double,  RowMajorEllpack, TNL::Devices::Host, long  >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     ColumnMajorEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorEllpack, TNL::Devices::Cuda, short >,
    TNL::Matrices::SparseMatrix< int,     ColumnMajorEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorEllpack, TNL::Devices::Cuda, int   >,
    TNL::Matrices::SparseMatrix< int,     ColumnMajorEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< long,    ColumnMajorEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< float,   ColumnMajorEllpack, TNL::Devices::Cuda, long  >,
    TNL::Matrices::SparseMatrix< double,  ColumnMajorEllpack, TNL::Devices::Cuda, long  >
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

TYPED_TEST( EllpackMatrixTest, setRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetRow< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_VectorProduct< EllpackMatrixType >();
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
