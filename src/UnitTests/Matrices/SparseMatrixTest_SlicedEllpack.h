/***************************************************************************
                          SparseMatrixTest_SlicedEllpack.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/SlicedEllpack.h>

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

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, short >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, short >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, short >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, short >,
    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
    ,TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( SlicedEllpackMatrixTest, SlicedEllpackMatrixTypes );

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
    
    test_SaveAndLoad< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, printTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_Print< SlicedEllpackMatrixType >();
}

#endif

#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
