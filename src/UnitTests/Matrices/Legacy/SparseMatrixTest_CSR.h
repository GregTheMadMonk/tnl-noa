/***************************************************************************
                          SparseMatrixTest_CSR.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/Legacy/CSR.h>

#include "SparseMatrixTest.hpp"
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
    TNL::Matrices::Legacy::CSR< int,    TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::CSR< long,   TNL::Devices::Host, int >,
   //  TNL::Matrices::Legacy::CSR< float,  TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::CSR< double, TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::CSR< int,    TNL::Devices::Host, long >,
    TNL::Matrices::Legacy::CSR< long,   TNL::Devices::Host, long >,
   //  TNL::Matrices::Legacy::CSR< float,  TNL::Devices::Host, long >,
    TNL::Matrices::Legacy::CSR< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
   ,TNL::Matrices::Legacy::CSR< int,    TNL::Devices::Cuda, int >,
   //  TNL::Matrices::Legacy::CSR< long,   TNL::Devices::Cuda, int >, // cuda atomicAdd has no support for long, only unsigned long long int
    TNL::Matrices::Legacy::CSR< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::CSR< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::CSR< int,    TNL::Devices::Cuda, long >,
   //  TNL::Matrices::Legacy::CSR< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::Legacy::CSR< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::Legacy::CSR< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( CSRMatrixTest, CSRMatrixTypes);

TYPED_TEST( CSRMatrixTest, setDimensionsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetDimensions< CSRMatrixType >();
}

//TYPED_TEST( CSRMatrixTest, setCompressedRowLengthsTest )
//{
////    using CSRMatrixType = typename TestFixture::CSRMatrixType;
//
////    test_SetCompressedRowLengths< CSRMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

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

TYPED_TEST( CSRMatrixTest, setElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetElement< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, addElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_AddElement< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setRowTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetRow< CSRMatrixType >();
}

/* TYPED_TEST( CSRMatrixTest, vectorProductTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_VectorProduct< CSRMatrixType >();
} */

/*TYPED_TEST( CSRMatrixTest, vectorProductLargerTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_VectorProductLarger< CSRMatrixType >();
}*/

TYPED_TEST( CSRMatrixTest, vectorProductCSRApadtiveTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_VectorProductCSRAdaptive< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, saveAndLoadTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SaveAndLoad< CSRMatrixType >( "test_SparseMatrixTest_CSR" );
}

TYPED_TEST( CSRMatrixTest, printTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_Print< CSRMatrixType >();
}

#endif

#include "../../main.h"
