/***************************************************************************
                          SparseMatrixTest_SlicedEllpack.h -  description
                             -------------------
    begin                : Dec 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>


#include "SparseMatrixTest.h"
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
template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, true, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, false, 32 >;

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, short, TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, short, TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >
#endif
>;

TYPED_TEST_SUITE( SlicedEllpackMatrixTest, SlicedEllpackMatrixTypes);

TYPED_TEST( SlicedEllpackMatrixTest, Constructors )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Constructors< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetDimensions< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setCompressedRowLengthsTest )
{
   using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

   test_SetCompressedRowLengths< SlicedEllpackMatrixType >();
}

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

TYPED_TEST( SlicedEllpackMatrixTest, getRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_GetRow< SlicedEllpackMatrixType >();
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

TYPED_TEST( SlicedEllpackMatrixTest, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, rowsReduction )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_RowsReduction< SlicedEllpackMatrixType >();
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
