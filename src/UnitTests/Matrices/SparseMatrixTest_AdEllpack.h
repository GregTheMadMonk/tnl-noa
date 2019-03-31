/***************************************************************************
                          SparseMatrixTest_AdEllpack.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/AdEllpack.h>

#include "SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class AdEllpackMatrixTest : public ::testing::Test
{
protected:
   using AdEllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using AdEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( AdEllpackMatrixTest, AdEllpackMatrixTypes);

// WORKING
TYPED_TEST( AdEllpackMatrixTest, setDimensionsTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetDimensions< AdEllpackMatrixType >();
}

//TYPED_TEST( AdEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
//    
////    test_SetCompressedRowLengths< AdEllpackMatrixType >();
//    
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

// WORKING
TYPED_TEST( AdEllpackMatrixTest, setLikeTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetLike< AdEllpackMatrixType, AdEllpackMatrixType >();
}

// WORKING
TYPED_TEST( AdEllpackMatrixTest, resetTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Reset< AdEllpackMatrixType >();
}

// SUPPOSEDLY WORKING - localLoad, offset and rowOffset are seemingly random numbers in the head and tail of WarpList.
TYPED_TEST( AdEllpackMatrixTest, setElementTest )
{
    // This test fails on m.setCompressedRowLengths( rowLengths ) in SparseMatrixTest.hpp
    
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetElement< AdEllpackMatrixType >();
}

// SUPPOSEDLY WORKING - localLoad, offset and rowOffset are seemingly random numbers in the head and tail of WarpList.
TYPED_TEST( AdEllpackMatrixTest, addElementTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_AddElement< AdEllpackMatrixType >();
}

// SUPPOSEDLY WORKING - Tests take longer than expected. setElement takes 13ms, compared to SlicedEllpack's 2ms.
TYPED_TEST( AdEllpackMatrixTest, setRowTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetRow< AdEllpackMatrixType >();
}

// SUPPOSEDLY WORKING
TYPED_TEST( AdEllpackMatrixTest, vectorProductTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_VectorProduct< AdEllpackMatrixType >();
}

// SUPPOSEDLY WORKING
TYPED_TEST( AdEllpackMatrixTest, saveAndLoadTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SaveAndLoad< AdEllpackMatrixType >( "test_SparseMatrixTest_AdEllpack" );
}

// SUPPOSEDLY WORKING
TYPED_TEST( AdEllpackMatrixTest, printTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Print< AdEllpackMatrixType >();
}

#ifdef NOT_WORKING
#endif

#endif


#include "../main.h"
