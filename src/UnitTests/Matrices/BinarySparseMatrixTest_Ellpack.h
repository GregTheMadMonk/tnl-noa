/***************************************************************************
                          BinarySparseMatrixTest_Ellpack.h -  description
                             -------------------
    begin                : Jan 30, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BinaryMatrixTest_Ellpack : public ::testing::Test
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
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack >
#endif
>;

TYPED_TEST_SUITE( BinaryMatrixTest_Ellpack, EllpackMatrixTypes);

TYPED_TEST( BinaryMatrixTest_Ellpack, setDimensionsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetDimensions< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, setCompressedRowLengthsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetCompressedRowLengths< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, setLikeTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetLike< EllpackMatrixType, EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, resetTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Reset< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, getRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_GetRow< EllpackMatrixType >();
}


TYPED_TEST( BinaryMatrixTest_Ellpack, setElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetElement< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_VectorProduct< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, rowsReduction )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_RowsReduction< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, saveAndLoadTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SaveAndLoad< EllpackMatrixType >( "test_BinarySparseMatrixTest_Ellpack" );
}

TYPED_TEST( BinaryMatrixTest_Ellpack, printTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Print< EllpackMatrixType >();
}

#endif

#include "../main.h"
