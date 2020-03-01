/***************************************************************************
                          BinarySparseMatrixTest_SlicedEllpack.h -  description
                             -------------------
    begin                : Jan 30, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BinaryMatrixTest_SlicedEllpack : public ::testing::Test
{
protected:
   using SlicedEllpackMatrixType = Matrix;
};

////
// Row-major format is used for the host system
template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, true, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, false, 32 >;

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >
#endif
>;

TYPED_TEST_SUITE( BinaryMatrixTest_SlicedEllpack, SlicedEllpackMatrixTypes);

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetDimensions< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setCompressedRowLengthsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetCompressedRowLengths< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setLikeTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetLike< SlicedEllpackMatrixType, SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, resetTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Reset< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, getRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_GetRow< SlicedEllpackMatrixType >();
}


TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, rowsReduction )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_RowsReduction< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, saveAndLoadTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SaveAndLoad< SlicedEllpackMatrixType >( "test_BinarySparseMatrixTest" );
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, printTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Print< SlicedEllpackMatrixType >();
}

#endif

#include "../main.h"
