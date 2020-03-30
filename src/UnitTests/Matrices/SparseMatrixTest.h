/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <iostream>
#include <sstream>

#include "SparseMatrixTest.hpp"

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE( MatrixTest, MatrixTypes);

TYPED_TEST( MatrixTest, Constructors )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Constructors< MatrixType >();
}

TYPED_TEST( MatrixTest, setDimensionsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetDimensions< MatrixType >();
}

TYPED_TEST( MatrixTest, setCompressedRowLengthsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetCompressedRowLengths< MatrixType >();
}

TYPED_TEST( MatrixTest, setLikeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST( MatrixTest, resetTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Reset< MatrixType >();
}

TYPED_TEST( MatrixTest, getRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetRow< MatrixType >();
}

TYPED_TEST( MatrixTest, setElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetElement< MatrixType >();
}

TYPED_TEST( MatrixTest, addElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddElement< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, rowsReduction )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_RowsReduction< MatrixType >();
}

TYPED_TEST( MatrixTest, saveAndLoadTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SaveAndLoad< MatrixType >( saveAndLoadFileName );
}

TYPED_TEST( MatrixTest, printTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Print< MatrixType >();
}

#endif