/***************************************************************************
                          BinarySparseMatrixTest.h -  description
                             -------------------
    begin                : Jan 30, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class CSRMatrixTest : public ::testing::Test
{
protected:
   using CSRMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using CSRMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, short, TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::BinaryMatrix, TNL::Containers::Segments::CSR >
#endif
>;

TYPED_TEST_SUITE( CSRMatrixTest, CSRMatrixTypes);

TYPED_TEST( CSRMatrixTest, setDimensionsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetDimensions< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setCompressedRowLengthsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetCompressedRowLengths< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setLikeTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetLike< CSRMatrixType, CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, resetTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_Reset< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, getRowTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_GetRow< CSRMatrixType >();
}


TYPED_TEST( CSRMatrixTest, setElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetElement< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, vectorProductTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_VectorProduct< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, rowsReduction )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_RowsReduction< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, saveAndLoadTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SaveAndLoad< CSRMatrixType >( "test_BinarySparseMatrixTest" );
}

TYPED_TEST( CSRMatrixTest, printTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_Print< CSRMatrixType >();
}

#endif

#include "../main.h"
