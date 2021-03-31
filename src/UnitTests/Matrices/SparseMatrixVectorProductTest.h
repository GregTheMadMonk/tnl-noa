/***************************************************************************
                          SparseMatrixVectorProductTest.h -  description
                             -------------------
    begin                : Mar 30, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
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

#include "SparseMatrixVectorProductTest.hpp"

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

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix1 )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_smallMatrix1< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix2 )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_smallMatrix2< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_smallMatrix3 )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_smallMatrix3< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix1 )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_mediumSizeMatrix1< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_mediumSizeMatrix2 )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_mediumSizeMatrix2< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_largeMatrix )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_largeMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest_longRowsMatrix )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct_longRowsMatrix< MatrixType >();
}

#endif
